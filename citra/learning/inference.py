"""Freeze -> encode -> index. Every certificate is bound to actual model bytes."""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from time import perf_counter
import numpy as np
import torch
from .fusion import LinearCorrection
from .gating import HistoryGate
from .aggregation import AggregationSpec

from citra import TemporalDocumentStore
from citra.kernel import immutable


def parameter_digest(model, prefix=None):
    digest = hashlib.sha256(json.dumps(model.config, sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        if prefix is None or name.startswith(prefix):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def synchronize(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class ForecastResult:
    prediction: np.ndarray
    certified_bound: object  # None for strategies not certified relative to Full Text
    aggregate_bound: object
    encoder_version: str
    impact_version: str
    index_generation: int
    timings: dict
    query_stats: object
    model_version: str
    session: object
    certificate_scope: str = "analytic_fixed_parameter_output_remainder_float64_tolerance"
    aggregation_semantics: object = None

    def assert_current(self):
        self.session.assert_current()
        if self.index_generation != getattr(self.session.store, "generation", 0) or self.model_version != self.session.model_version:
            raise RuntimeError("stale result: index/model generation changed")


class FrozenForecastSession:
    @staticmethod
    def validate_fusion(model):
        if (type(model.fusion) is not LinearCorrection
                or type(model.fusion.projection) is not torch.nn.Linear
                or model.fusion.projection.bias is not None
                or any(m._forward_hooks or m._forward_pre_hooks for m in model.fusion.modules())):
            raise ValueError("only the built-in linear correction without modifying hooks has a supported certificate")
        if model.gate is not None and (type(model.gate) is not HistoryGate
                or type(model.gate.projection) is not torch.nn.Linear
                or any(m._forward_hooks or m._forward_pre_hooks for m in model.gate.modules())):
            raise ValueError("only the built-in history-only bounded gate is supported")

    def __init__(self, model, corpus=None, *, text_enabled=True, build_index=True):
        self.text_enabled = text_enabled
        if text_enabled:
            self.validate_fusion(model)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("freeze parameters with eval() and requires_grad_(False) before building an index")
        if any(p.dtype != torch.float64 for p in model.parameters()):
            raise ValueError("this certificate implementation requires float64 parameters")
        self.model, self.corpus = model, corpus if text_enabled else None
        self.spec=AggregationSpec.from_model(model)
        self.corpus_version = corpus.fingerprint if text_enabled else None
        self.model_version = parameter_digest(model, None if text_enabled else "backbone.")
        self.offline_timings = dict(index_construction_seconds=0., index_maintenance_seconds=None)
        self.store = None
        self.index_generation = 0
        self.encoder_version = self.impact_version = None
        if not text_enabled:
            self.offline_timings["index_maintenance_seconds"] = 0.
            self.offline_timings.update(base_feature_transfer_seconds=0., foundation_text_encoding_seconds=0.,
                text_key_projection_seconds=0., impact_encoding_seconds=0., encoded_document_transfer_seconds=0.,
                metadata_store_construction_seconds=0.)
            return
        self.encoder_version = hashlib.sha256((corpus.base_encoder_version +
            parameter_digest(model, "key_encoder.") + parameter_digest(model, "query_encoder.")).encode()).hexdigest()
        self.impact_version = parameter_digest(model, "impact_encoder.")
        synchronize(model)
        start = perf_counter()
        self.base, self.metadata = corpus.tensors(next(model.parameters()).device, torch.float64)
        synchronize(model)
        self.offline_timings["base_feature_transfer_seconds"] = perf_counter() - start
        self.offline_timings["foundation_text_encoding_seconds"] = None  # supplied as offline data, not measured here
        start = perf_counter()
        with torch.no_grad():
            keys = model.key_encoder(self.base)
        synchronize(model)
        self.offline_timings["text_key_projection_seconds"] = perf_counter() - start
        start = perf_counter()
        with torch.no_grad():
            impacts = model.impact_encoder(self.base, self.metadata)
        synchronize(model)
        self.offline_timings["impact_encoding_seconds"] = perf_counter() - start
        start = perf_counter()
        keys, impacts = keys.cpu().numpy(), impacts.cpu().numpy()
        self.keys_np, self.impacts_np = immutable(keys), immutable(impacts)
        self._amplification = float(np.linalg.norm(model.fusion.projection.weight.detach().cpu().numpy()))
        self.offline_timings["encoded_document_transfer_seconds"] = perf_counter() - start
        start = perf_counter()
        c = model.config
        self.store = TemporalDocumentStore(c["embedding_dim"], c["impact_dim"], sigma=c["sigma"],
                                           allow_simulated=corpus.allow_simulated, indexed=False)
        seen = set()
        for d, z, v in zip(corpus.documents, keys, impacts):
            encoded = replace(d, embedding=z, impact_vector=v)
            encoded = self.spec.indexed_document(encoded)
            if encoded is None:
                continue
            (self.store.append_revision if d.doc_id in seen else self.store.insert_document)(encoded)
            seen.add(d.doc_id)
        self.index_generation = self.store.generation
        self.offline_timings["metadata_store_construction_seconds"] = perf_counter() - start
        if build_index:
            self.prepare_index()
        self.assert_current()

    def prepare_index(self):
        self.assert_current()
        if not self.text_enabled:
            raise ValueError("TS-only execution does not need an index")
        elapsed = self.store.build_index()
        self.offline_timings["index_construction_seconds"] += elapsed
        return elapsed

    def assert_current(self):
        if not self.text_enabled:
            if (self.model.training or any(p.requires_grad for p in self.model.backbone.parameters())
                    or parameter_digest(self.model, "backbone.") != self.model_version):
                raise RuntimeError("numeric model parameters changed")
            return
        self.validate_fusion(self.model)
        if (self.model.training or any(p.requires_grad for p in self.model.parameters())
                or parameter_digest(self.model) != self.model_version
                or self.corpus.fingerprint != self.corpus_version
                or self.store.generation != self.index_generation):
            raise RuntimeError("stale frozen index: explicitly freeze/re-encode/rebuild after model or corpus updates")

    @torch.no_grad()
    def predict(self, history, cutoff, epsilon=0., execution="citra_certified", topk=8):
        if execution not in ("ts_only", "semantic_topk", "full_asof_text", "citra_certified"):
            raise ValueError("invalid inference mode")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        model = self.model
        synchronize(model)
        total_start = perf_counter()
        start = perf_counter()
        self.assert_current()
        timings = {"version_validation_seconds": perf_counter() - start}
        start = perf_counter()
        x = torch.as_tensor(history, dtype=torch.float64, device=next(model.parameters()).device)
        synchronize(model)
        timings["input_transfer_seconds"] = perf_counter() - start
        start = perf_counter()
        c = model.config
        if x.shape != (1, c["seq_len"], c["channels"]) or not torch.isfinite(x).all():
            raise ValueError("one forecast requires finite history [1,L,C]")
        synchronize(model)
        timings["input_validation_seconds"] = perf_counter() - start
        if execution == "ts_only":
            start = perf_counter()
            prediction = model.numeric_forecast(x)
            synchronize(model)
            timings["forecast_inference_seconds"] = perf_counter() - start
            start = perf_counter()
            prediction = immutable(prediction.cpu().numpy())
            timings["output_transfer_seconds"] = perf_counter() - start
            timings.update(query_encoding_seconds=0., index_query_seconds=0., exact_aggregation_seconds=0.,
                approximate_aggregation_seconds=0., aggregate_transfer_seconds=0., query_transfer_seconds=0.,
                certificate_setup_seconds=0.,gate_encoding_seconds=0.)
            timings["data_transfer_seconds"] = timings["input_transfer_seconds"] + timings["output_transfer_seconds"]
            timings["total_inference_seconds"] = perf_counter() - total_start
            return ForecastResult(prediction, None, None, self.encoder_version, self.impact_version,
                self.index_generation, timings, None, self.model_version, self, "ts_only_no_text_certificate")
        if not self.text_enabled:
            raise ValueError("this session is numeric-only")
        start = perf_counter()
        query_tensor = model.encode_query(x)
        synchronize(model)
        timings["query_encoding_seconds"] = perf_counter() - start
        gate=None
        gate_number=1.
        timings["gate_encoding_seconds"]=0.
        if model.gate is not None:
            start=perf_counter()
            gate=model.gate_value(x)
            gate_number=float(gate.item())
            timings["gate_encoding_seconds"]=perf_counter()-start
        start = perf_counter()
        query = query_tensor.cpu().numpy()[0]
        timings["query_transfer_seconds"] = perf_counter() - start
        start = perf_counter()
        amplification = self._amplification*gate_number
        timings["certificate_setup_seconds"] = perf_counter() - start
        start = perf_counter()
        stats, aggregate_bound = None, 0.
        mask=self.spec.mask(self.corpus,[cutoff])[0]
        count=int(mask.sum())
        denominator=self.spec.denominator(count)
        if execution == "ts_only":
            aggregate = np.zeros(c["impact_dim"])
        elif execution == "citra_certified":
            if self.store.index is None:
                raise ValueError("call prepare_index() offline before measuring certified queries")
            budget = 0. if epsilon == 0 else (np.finfo(np.float64).max if amplification == 0 else
                     max(0., float(np.nextafter(min(epsilon / amplification*denominator, np.finfo(np.float64).max), 0.))))
            result = self.store.query_certified(query, cutoff, budget)
            if amplification * (result.aggregate_bound/denominator) > epsilon:
                result = self.store.query_certified(query, cutoff, 0.)
            aggregate, aggregate_bound, stats = result.aggregate, result.aggregate_bound, result.stats
        elif execution == "full_asof_text":
            z, v = self.keys_np[mask], self.impacts_np[mask]
            weights = np.exp(-np.sum((query-z)**2, axis=1)/(2*c["sigma"]**2))
            aggregate = np.sum(weights[:, None]*v, axis=0)
        else:
            if topk <= 0:
                raise ValueError("topk must be positive")
            z, v = self.keys_np[mask], self.impacts_np[mask]
            distances = np.sum((query-z)**2, axis=1)
            k = min(topk, len(z))
            chosen = np.argpartition(distances, k-1)[:k] if k else np.array([], dtype=int)
            weights = np.exp(-distances[chosen]/(2*c["sigma"]**2))
            aggregate = np.sum(weights[:, None]*v[chosen], axis=0)
        aggregate=aggregate/denominator
        aggregate_bound=aggregate_bound/denominator
        timings["index_query_seconds"] = perf_counter() - start
        timings["exact_aggregation_seconds"] = timings["index_query_seconds"] if execution != "citra_certified" else 0.
        timings["approximate_aggregation_seconds"] = timings["index_query_seconds"] if execution == "citra_certified" else 0.
        start = perf_counter()
        a = torch.tensor(aggregate.copy(), dtype=torch.float64, device=x.device)[None]
        synchronize(model)
        timings["aggregate_transfer_seconds"] = perf_counter() - start
        start = perf_counter()
        numerical = model.numeric_forecast(x)
        correction=model.fusion(a)
        prediction = numerical + (correction if gate is None else gate[:,:,None]*correction)
        synchronize(model)
        timings["forecast_inference_seconds"] = perf_counter() - start
        start = perf_counter()
        prediction = immutable(prediction.cpu().numpy())
        timings["output_transfer_seconds"] = perf_counter() - start
        timings["data_transfer_seconds"] = sum(timings[key] for key in (
            "input_transfer_seconds", "query_transfer_seconds", "aggregate_transfer_seconds", "output_transfer_seconds"))
        timings["total_inference_seconds"] = perf_counter() - total_start
        bound = amplification * aggregate_bound if execution in ("citra_certified", "full_asof_text") else None
        scope = ("uncertified_topk_relative_to_full" if execution == "semantic_topk" else
                 "ts_only_no_text_certificate" if execution == "ts_only" else
                 "analytic_fixed_parameter_output_remainder_float64_tolerance")
        return ForecastResult(prediction, bound, aggregate_bound if bound is not None else None, self.encoder_version, self.impact_version,
                              self.index_generation, timings, stats, self.model_version, self, scope,
                              dict(kind=self.spec.kind,window_days=self.spec.window_days,visible_documents=count,
                                   denominator=denominator,gate=gate_number,effective_input_lipschitz=amplification,
                                   effective_raw_sum_lipschitz=amplification/denominator))
