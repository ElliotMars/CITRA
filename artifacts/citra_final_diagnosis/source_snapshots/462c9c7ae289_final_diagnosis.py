"""Final diagnosis helpers: input interventions, shared metadata and paired inference."""
import copy
import hashlib
import json
from datetime import timedelta
import numpy as np
from data_provider.citra_weather import TextCorpus,epoch_seconds,datetime_utc


PUBLIC_FIELDS=("doc_id","version_id","source","entity_id","office_id","source_url",
    "issued_at","available_at","valid_from","valid_to","event_start","event_end",
    "ingested_at","availability_provenance","availability_protocol","sha256","retrieved_at")


def control_view(data,kind):
    if kind not in ("metadata_only","content_only","constant_text","null_no_metadata"):
        raise ValueError("unknown fixed content intervention")
    records=[]
    for source in data.corpus.records:
        # A whitelist, not dict(source): non-content controls never access the
        # source embedding or text. Hashes/identities only preserve provenance.
        row={key:source[key] for key in PUBLIC_FIELDS if key in source}
        vector=np.zeros(data.corpus.base_dim)
        if kind=="content_only":vector=np.asarray(source["embedding"],dtype=float)
        elif kind=="constant_text":vector[0]=1.
        row.update(embedding=vector.tolist(),metadata=list(source["metadata"]) if kind in
            ("metadata_only","constant_text") else [0.]*data.corpus.metadata_dim,
            text="[content withheld by registered control]",input_control=kind)
        records.append(row)
    result=copy.copy(data)
    version=data.corpus.base_encoder_version+":metadata-zero" if kind=="content_only" else "no-content-input:"+kind
    result.corpus=TextCorpus(records,data.corpus.base_dim,data.corpus.metadata_dim,
        allow_simulated=data.corpus.allow_simulated,base_encoder_version=version)
    result.control_kind=kind
    result.manifest=dict(data.manifest,input_control=kind,base_encoder_version=version)
    if kind!="content_only":result.manifest.pop("frozen_text_encoder",None)
    result.fingerprint=hashlib.sha256((data.fingerprint+result.corpus.fingerprint+kind).encode()).hexdigest()
    return result


class SharedWindowMetadata:
    """Plain vector arrays cached offline for EVERY benchmark method, no new index."""
    def __init__(self,corpus,days=7):
        self.corpus=corpus
        duration=timedelta(days=days)
        self.duration_us=(duration.days*86400+duration.seconds)*1000000+duration.microseconds
        self.available=np.array([epoch_seconds(d.available_at)*1000000+d.available_at.microsecond
            if d.available_at is not None else np.iinfo(np.int64).max for d in corpus.documents],dtype=np.int64)
        self.bytes=self.available.nbytes+corpus._starts.nbytes+corpus._ends.nbytes

    def mask(self,cutoff):
        if isinstance(cutoff,(int,np.integer)):cutoff=datetime_utc(int(cutoff))
        t=epoch_seconds(cutoff)*1000000+cutoff.microsecond
        return self.corpus.visibility([cutoff])[0] & (self.available>t-self.duration_us)


def describe(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),p50=float(np.percentile(a,50)),
        p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=float(a.max()),min=float(a.min()))


def paired_block_bootstrap(dates,seed_date_differences,block_days=7,repeats=2000,seed=54321):
    dates=np.asarray(dates,dtype="datetime64[D]")
    differences=np.asarray(seed_date_differences,dtype=float)
    if differences.ndim!=2 or differences.shape[1]!=len(dates) or not np.isfinite(differences).all():
        raise ValueError("expected finite [training seeds, common forecast dates] paired losses")
    if len(np.unique(dates))!=len(dates):raise ValueError("dates must be unique, never flatten training seeds")
    years=dates.astype("datetime64[Y]")
    if len(np.unique(years))!=1:raise ValueError("bootstrap each calendar year separately")
    start=years[0].astype("datetime64[D]")
    end=(years[0]+np.timedelta64(1,"Y")).astype("datetime64[D]")
    length=int((end-start)/np.timedelta64(1,"D"))
    dense=np.full(length,np.nan)
    daily=differences.mean(0)
    dense[((dates-start)/np.timedelta64(1,"D")).astype(int)]=daily
    rng=np.random.default_rng(seed)
    blocks=rng.integers(0,length,size=(repeats,int(np.ceil(length/block_days))))
    indices=((blocks[:,:,None]+np.arange(block_days))%length).reshape(repeats,-1)[:,:length]
    draws=np.nanmean(dense[indices],axis=1)
    return dict(mean_difference=float(daily.mean()),ci95_low=float(np.percentile(draws,2.5)),
        ci95_high=float(np.percentile(draws,97.5)),improvement_date_fraction=float(np.mean(daily<0)),
        seed_mean_differences=differences.mean(1).tolist(),dates=len(dates),training_seeds=len(differences),
        block_days=block_days,repeats=repeats,seed=seed,
        unit="one calendar date, paired loss averaged over training seeds before resampling",
        method="circular calendar-day moving blocks; missing forecast dates remain NaN; unadjusted percentile CI")
