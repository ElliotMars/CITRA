"""Real-data protocol semantics and mode-specific accounting, using offline fixtures."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from citra.availability import issuance_records,ASSUMED,VERIFIED
from data_provider.citra_weather import make_fixture,WeatherData,parse_time,datetime_utc
from scripts.encode_citra_text import load_features


class ReplayProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="citra-real-protocol-")
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        make_fixture(self.root/"data",hours=180,base_dim=8)

    def test_issuance_delay_is_not_verified(self):
        original=json.loads((self.root/"data"/"documents.jsonl").read_text().splitlines()[2])
        for delay in (0,1,6,24):
            records,audit=issuance_records([original],delay,parse_time)
            row=records[0]
            self.assertEqual(row["availability_provenance"],"simulated")
            self.assertEqual(row["availability_protocol"],ASSUMED)
            self.assertEqual(parse_time(row["available_at"])-parse_time(row["issued_at"]),timedelta(hours=delay))
            self.assertFalse(audit["historical_first_availability_verified"])

    def test_ambiguous_correction_order_is_excluded(self):
        row=json.loads((self.root/"data"/"documents.jsonl").read_text().splitlines()[0])
        records,audit=issuance_records([row,dict(row,version_id="uncertain-correction")],0,parse_time)
        self.assertEqual(records,[])
        self.assertEqual(audit["records_excluded"],2)

    def test_delay_does_not_change_numeric_cohort_or_scaler(self):
        a=WeatherData(self.root/"data",availability_protocol=ASSUMED,text_delay_hours=0)
        b=WeatherData(self.root/"data",availability_protocol=ASSUMED,text_delay_hours=24)
        np.testing.assert_array_equal(a.mean,b.mean)
        np.testing.assert_array_equal(a.scale,b.scale)
        self.assertEqual(a.numeric_fingerprint,b.numeric_fingerprint)
        self.assertEqual(a.windows("test",12,3,[0,1]).starts,b.windows("test",12,3,[0,1]).starts)
        self.assertNotEqual(a.corpus.fingerprint,b.corpus.fingerprint)

    def test_compiled_visibility_matches_independent_version_oracle(self):
        corpus=WeatherData(self.root/"data",allow_simulated=True).corpus
        origin=parse_time(corpus.records[0]["issued_at"])
        for hour in (-1,0,1,9,24,72,180):
            cutoff=origin+timedelta(hours=hour,microseconds=1)
            expected={(d.doc_id,d.version_id) for d in corpus.store.snapshot(cutoff)}
            mask=corpus.visibility([cutoff])[0]
            actual={(d.doc_id,d.version_id) for d,keep in zip(corpus.documents,mask) if keep}
            self.assertEqual(actual,expected)

    def test_causal_control_never_imports_future_donors(self):
        data=WeatherData(self.root/"data",allow_simulated=True)
        control=data.with_causal_shuffle(19)
        self.assertEqual(data.numeric_fingerprint,control.numeric_fingerprint)
        changed=0
        for row,source in zip(control.corpus.records,data.corpus.documents):
            donor=row["control_donor_available_at"]
            activation=max(t for t in (source.available_at,source.valid_from,source.ingested_at) if t is not None)
            if donor is not None:
                self.assertLessEqual(parse_time(donor),activation)
            changed+=row["control_donor_doc_id"]!=source.doc_id
        self.assertGreater(changed,0)
        np.testing.assert_array_equal(data.corpus.visibility([data.times[70]]),control.corpus.visibility([data.times[70]]))

    def test_offline_feature_identity_is_content_bound(self):
        root=self.root/"features"
        root.mkdir()
        records=[dict(doc_id="one",version_id="v1",text="A fixed input")]
        np.savez_compressed(root/"embeddings.npz",vectors=np.array([[1.,2.]],dtype=np.float32))
        row=dict(doc_id="one",version_id="v1",vector_index=0,text_sha256=hashlib.sha256(records[0]["text"].encode()).hexdigest())
        (root/"identities.jsonl").write_text(json.dumps(row)+"\n")
        manifest={name:hashlib.sha256((root/file).read_bytes()).hexdigest() for name,file in (
            ("embeddings_sha256","embeddings.npz"),("identities_sha256","identities.jsonl"))}
        (root/"manifest.json").write_text(json.dumps(manifest))
        result,_=load_features(root,records)
        self.assertEqual(result[0]["embedding"],[1.,2.])
        with self.assertRaises(ValueError):
            load_features(root,[dict(records[0],text="A changed input")])

    def test_verified_protocol_rejects_simulated_numeric_data(self):
        with self.assertRaises(ValueError):
            WeatherData(self.root/"data",availability_protocol=VERIFIED,allow_simulated=True)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"usable torch is required")
class AccountingTests(unittest.TestCase):
    def test_decomposition_numeric_backbone_trains_and_respects_targets(self):
        from models.CITRA import Model
        model=Model(seq_len=12,channels=3,horizon=4,target_ids=[2,0],base_dim=8,backbone_kind="dlinear").double()
        x=torch.randn(2,12,3,dtype=torch.float64,requires_grad=True)
        y=model.numeric_forecast(x)
        self.assertEqual(y.shape,(2,4,2))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is not None for p in model.backbone.parameters()))

    def test_numeric_session_does_no_text_work(self):
        from models.CITRA import Model
        from citra.learning.inference import FrozenForecastSession
        model=Model(seq_len=12,channels=2,horizon=3,base_dim=8).double().eval().requires_grad_(False)
        with patch("citra.learning.inference.TemporalDocumentStore",side_effect=AssertionError("unnecessary text store")):
            session=FrozenForecastSession(model,text_enabled=False,build_index=False)
        with patch.object(model,"encode_query",side_effect=AssertionError("unnecessary text query")):
            result=session.predict(np.zeros((1,12,2)),datetime_utc(1577836800),execution="ts_only")
        self.assertIsNone(session.store)
        self.assertIsNone(result.certified_bound)
        self.assertEqual(result.timings["query_encoding_seconds"],0.)
        self.assertEqual(session.offline_timings["index_construction_seconds"],0.)

    def test_exact_session_builds_index_only_when_requested(self):
        from models.CITRA import Model
        from citra.learning.inference import FrozenForecastSession
        with tempfile.TemporaryDirectory() as tmp:
            make_fixture(Path(tmp)/"data",hours=180,base_dim=8)
            data=WeatherData(Path(tmp)/"data",allow_simulated=True)
            model=Model(seq_len=12,channels=2,horizon=3,base_dim=8).double().eval().requires_grad_(False)
            session=FrozenForecastSession(model,data.corpus,build_index=False)
            x,_,cutoff=data.windows("test",12,3,[0,1])[0]
            exact=session.predict(x[None],datetime_utc(cutoff),execution="full_asof_text")
            self.assertIsNone(session.store.index)
            session.prepare_index()
            approximate=session.predict(x[None],datetime_utc(cutoff),.1)
            self.assertLessEqual(np.linalg.norm(exact.prediction-approximate.prediction),approximate.certified_bound+1e-10)
