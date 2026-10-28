"""Freeze -> encode -> index. Every certificate is bound to actual model bytes."""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from time import perf_counter
import numpy as np
import torch
from .fusion import LinearCorrection

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

    def assert_current(self):
        self.session.assert_current()
        if self.index_generation != self.session.store.generation or self.model_version != self.session.model_version:
            raise RuntimeError("stale result: index/model generation changed")


class FrozenForecastSession:
    @staticmethod
    def validate_fusion(model):
        if (type(model.fusion) is not LinearCorrection
                or type(model.fusion.projection) is not torch.nn.Linear
                or model.fusion.projection.bias is not None
                or any(m._forward_hooks or m._forward_pre_hooks for m in model.fusion.modules())):
            raise ValueError("only the built-in linear correction without modifying hooks has a supported certificate")

    def __init__(self, model, corpus):
        self.validate_fusion(model)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("freeze parameters with eval() and requires_grad_(False) before building an index")
        if any(p.dtype != torch.float64 for p in model.parameters()):
            raise ValueError("this certificate implementation requires float64 parameters")
        self.model, self.corpus = model, corpus
        self.corpus_version = corpus.fingerprint
        self.model_version = parameter_digest(model)
        self.encoder_version = hashlib.sha256((corpus.base_encoder_version +
            parameter_digest(model, "key_encoder.") + parameter_digest(model, "query_encoder.")).encode()).hexdigest()
        self.impact_version = parameter_digest(model, "impact_encoder.")
        self.offline_timings = {}
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
        self.offline_timings["encoded_document_transfer_seconds"] = perf_counter() - start
        start = perf_counter()
        c = model.config
        self.store = TemporalDocumentStore(c["embedding_dim"], c["impact_dim"], sigma=c["sigma"],
                                           allow_simulated=corpus.allow_simulated)
        seen = set()
        for d, z, v in zip(corpus.documents, keys, impacts):
            encoded = replace(d, embedding=z, impact_vector=v)
            (self.store.append_revision if d.doc_id in seen else self.store.insert_document)(encoded)
            seen.add(d.doc_id)
        self.index_generation = self.store.generation
        self.offline_timings["index_construction_seconds"] = perf_counter() - start
        self.assert_current()

    def assert_current(self):
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
        x = torch.as_tensor(history, dtype=torch.float64, device=next(model.parameters()).device)
        c = model.config
        if x.shape != (1, c["seq_len"], c["channels"]) or not torch.isfinite(x).all():
            raise ValueError("one forecast requires finite history [1,L,C]")
        synchronize(model)
        start = perf_counter()
        query = model.encode_query(x).cpu().numpy()[0]
        synchronize(model)
        timings["query_encoding_seconds"] = perf_counter() - start
        start = perf_counter()
        amplification = float(np.linalg.norm(model.fusion.projection.weight.detach().cpu().numpy()))
        timings["certificate_setup_seconds"] = perf_counter() - start
        start = perf_counter()
        stats, aggregate_bound = None, 0.
        if execution == "ts_only":
            aggregate = np.zeros(c["impact_dim"])
        elif execution == "citra_certified":
            budget = 0. if epsilon == 0 else (np.finfo(np.float64).max if amplification == 0 else
                     max(0., float(np.nextafter(min(epsilon / amplification, np.finfo(np.float64).max), 0.))))
            result = self.store.query_certified(query, cutoff, budget)
            if amplification * result.aggregate_bound > epsilon:
                result = self.store.query_certified(query, cutoff, 0.)
            aggregate, aggregate_bound, stats = result.aggregate, result.aggregate_bound, result.stats
        elif execution == "full_asof_text":
            aggregate = self.store.aggregate_exact(query, cutoff)
        else:
            if topk <= 0:
                raise ValueError("topk must be positive")
            docs = self.store.snapshot(cutoff)
            chosen = sorted(docs, key=lambda d: float(np.sum((query - d.embedding)**2)))[:topk]
            from citra.kernel import exact_aggregate
            aggregate = exact_aggregate(chosen, query, self.store.kernel, c["impact_dim"])
        timings["index_query_seconds"] = perf_counter() - start
        start = perf_counter()
        a = torch.tensor(aggregate.copy(), dtype=torch.float64, device=x.device)[None]
        synchronize(model)
        timings["aggregate_transfer_seconds"] = perf_counter() - start
        start = perf_counter()
        numerical = model.numeric_forecast(x)
        prediction = numerical if execution == "ts_only" else numerical + model.fusion(a)
        synchronize(model)
        prediction = immutable(prediction.cpu().numpy())
        timings["forecast_inference_seconds"] = perf_counter() - start
        timings["total_inference_seconds"] = perf_counter() - total_start
        bound = amplification * aggregate_bound if execution in ("citra_certified", "full_asof_text") else None
        scope = ("uncertified_topk_relative_to_full" if execution == "semantic_topk" else
                 "ts_only_no_text_certificate" if execution == "ts_only" else
                 "analytic_fixed_parameter_output_remainder_float64_tolerance")
        return ForecastResult(prediction, bound, aggregate_bound if bound is not None else None, self.encoder_version, self.impact_version,
                              self.index_generation, timings, stats, self.model_version, self, scope)
