"""Explicit query operators and an independent compensated exact oracle."""
from dataclasses import dataclass,replace
from datetime import timedelta
import math
import numpy as np
from citra.kernel import RBFKernel,exact_aggregate
from data_provider.citra_weather import datetime_utc,epoch_seconds


@dataclass(frozen=True)
class AggregationSpec:
    kind: str = "sum"
    window_days: object = None

    def __post_init__(self):
        if self.kind not in ("sum","count_mean"):
            raise ValueError("aggregation kind must be sum or count_mean")
        if self.window_days is not None and (not math.isfinite(self.window_days) or self.window_days<=0):
            raise ValueError("availability window must have finite positive days")

    @classmethod
    def from_model(cls,model):
        return cls(model.config.get("aggregation_kind","sum"),model.config.get("window_days"))

    def mask(self,corpus,cutoffs):
        mask=corpus.visibility(cutoffs)
        if self.window_days is not None:
            times=[datetime_utc(int(t)) if isinstance(t,(int,np.integer)) else t for t in cutoffs]
            # Public availability, not event time; (cutoff-window, cutoff].
            duration=timedelta(days=self.window_days)
            duration_us=(duration.days*86400+duration.seconds)*1000000+duration.microseconds
            lower=np.array([epoch_seconds(t)*1000000+t.microsecond-duration_us for t in times],dtype=np.int64)
            available=np.array([epoch_seconds(d.available_at)*1000000+d.available_at.microsecond
                if d.available_at is not None else np.iinfo(np.int64).max for d in corpus.documents],dtype=np.int64)
            mask&=available[None]>lower[:,None]
        return mask

    def denominator(self,count):
        return max(int(count),1) if self.kind=="count_mean" else 1

    def indexed_document(self,document):
        if self.window_days is None or document.available_at is None:
            return document
        expires=document.available_at+timedelta(days=self.window_days)
        if document.valid_to is not None:
            expires=min(expires,document.valid_to)
        return None if expires<=document.valid_from else replace(document,valid_to=expires)


def exact_oracle(corpus,keys,impacts,query,cutoff,sigma,spec):
    """Snapshot/version sweep + per-document fsum, independent of mask/tree."""
    docs=corpus.store.snapshot(cutoff)
    if spec.window_days is not None:
        lower=cutoff-timedelta(days=spec.window_days)
        docs=tuple(d for d in docs if d.available_at>lower)
    positions={(d.doc_id,d.version_id):i for i,d in enumerate(corpus.documents)}
    encoded=[]
    for d in docs:
        i=positions[d.doc_id,d.version_id]
        encoded.append(replace(d,embedding=keys[i],impact_vector=impacts[i]))
    denominator=spec.denominator(len(docs))
    value=exact_aggregate(encoded,query,RBFKernel(sigma),impacts.shape[1])/denominator
    return value,len(docs)
