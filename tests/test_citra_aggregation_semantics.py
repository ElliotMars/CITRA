"""Independent semantic oracles, frozen residuals and effective gated bounds."""
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
import hashlib
import json
import numpy as np
import torch

from citra.learning.aggregation import AggregationSpec,exact_oracle
from citra.learning.inference import FrozenForecastSession
from citra.learning.training import batch_tensors
from data_provider.citra_weather import make_fixture,WeatherData,datetime_utc
from models.CITRA import Model


class AggregationSemanticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(12)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        make_fixture(Path(self.tmp.name)/"data",hours=240,base_dim=8)
        self.data=WeatherData(Path(self.tmp.name)/"data",allow_simulated=True)
        self.windows=self.data.windows("test",12,3,[0,1],24)

    def model(self,**kw):
        return Model(seq_len=12,channels=2,horizon=3,base_dim=8,**kw).double()

    def test_mean_divides_by_visible_count_not_weight_sum(self):
        m=self.model(aggregation_kind="count_mean",sigma=1.)
        q=torch.tensor([[1.,0.]],dtype=torch.float64)
        z=torch.tensor([[1.,0.],[-1.,0.],[1.,0.]],dtype=torch.float64)
        v=torch.tensor([[2.,-1.],[4.,3.],[100.,100.]],dtype=torch.float64)
        mask=torch.tensor([[True,True,False]])
        actual=m.aggregate(q,z,v,mask)
        expected=(v[0]+np.exp(-2)*v[1])/2
        torch.testing.assert_close(actual[0],expected)

    def test_window_lower_boundary_is_exclusive(self):
        corpus=self.data.corpus
        d=corpus.documents[0]
        spec=AggregationSpec("sum",.25)
        t=d.available_at+timedelta(hours=6)
        self.assertFalse(spec.mask(corpus,[t])[0,0])
        self.assertTrue(spec.mask(corpus,[t-timedelta(microseconds=1)])[0,0])
        micro=AggregationSpec("sum",1/(86400*1000000))
        self.assertFalse(micro.mask(corpus,[d.available_at+timedelta(microseconds=1)])[0,0])
        self.assertTrue(micro.mask(corpus,[d.available_at])[0,0])

    def test_exact_and_approximate_match_independent_oracle(self):
        for kind,days,gated in (("sum",None,False),("count_mean",None,False),
                ("sum",7,False),("sum",30,False),("sum",90,False),
                ("count_mean",.25,False),("count_mean",.25,True)):
            with self.subTest(kind=kind,days=days,gated=gated):
                m=self.model(aggregation_kind=kind,window_days=days,gated=gated).eval().requires_grad_(False)
                session=FrozenForecastSession(m,self.data.corpus)
                x,_,t=self.windows[0];cutoff=datetime_utc(t)
                q=m.encode_query(torch.tensor(x[None])).detach().numpy()[0]
                a,n=exact_oracle(self.data.corpus,session.keys_np,session.impacts_np,q,cutoff,m.config["sigma"],session.spec)
                tx=torch.tensor(x[None])
                expected=(m.numeric_forecast(tx)+m.text_correction(tx,torch.tensor(a[None]))).detach().numpy()
                exact=session.predict(x[None],cutoff,execution="full_asof_text")
                np.testing.assert_allclose(exact.prediction,expected,atol=1e-10,rtol=1e-10)
                for epsilon in (0.,.01,1.):
                    fast=session.predict(x[None],cutoff,epsilon)
                    self.assertLessEqual(np.linalg.norm(fast.prediction-expected),fast.certified_bound+1e-10)
                    self.assertLessEqual(fast.certified_bound,epsilon)
                    self.assertEqual(fast.aggregation_semantics["visible_documents"],n)
                    self.assertEqual(fast.aggregation_semantics["denominator"],max(n,1) if kind=="count_mean" else 1)

    def test_gate_bound_uses_actual_history_and_norm(self):
        m=self.model(gated=True,aggregation_kind="count_mean").eval().requires_grad_(False)
        m.gate.projection.bias.fill_(1.25)
        session=FrozenForecastSession(m,self.data.corpus)
        x,_,t=self.windows[0]
        r=session.predict(x[None],datetime_utc(t),.1)
        info=r.aggregation_semantics
        expected=float(torch.sigmoid(torch.tensor(1.25,dtype=torch.float64)))*float(m.fusion.lipschitz_bound())
        self.assertAlmostEqual(info["effective_input_lipschitz"],expected,places=12)
        self.assertAlmostEqual(info["effective_raw_sum_lipschitz"],expected/info["denominator"],places=12)
        m.gate.projection.bias.add_(.1)
        with self.assertRaises(RuntimeError):session.predict(x[None],datetime_utc(t),.1)

    def test_unsupported_gate_cannot_claim_certificate(self):
        m=self.model(gated=True).eval().requires_grad_(False)
        m.gate=torch.nn.Sequential(torch.nn.Linear(24,1),torch.nn.Sigmoid()).double().requires_grad_(False)
        with self.assertRaises(ValueError):FrozenForecastSession(m,self.data.corpus)

    def test_zero_head_residual_starts_at_ts_and_never_updates_backbone(self):
        ts=self.model().eval();m=self.model(gated=True,aggregation_kind="count_mean",window_days=7)
        m.backbone.load_state_dict(ts.backbone.state_dict());m.backbone.requires_grad_(False)
        torch.nn.init.zeros_(m.fusion.projection.weight)
        old={k:v.detach().clone() for k,v in m.backbone.state_dict().items()}
        x,y,mask=batch_tensors(self.windows,[0,1],"cpu",aggregation_spec=AggregationSpec.from_model(m))
        base,meta=self.data.corpus.tensors("cpu",torch.float64)
        torch.testing.assert_close(m(x,base,meta,mask),ts.numeric_forecast(x),atol=0,rtol=0)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        for _ in range(3):
            opt.zero_grad();(m(x,base,meta,mask)-y).square().mean().backward();opt.step()
        for k,v in m.backbone.state_dict().items():torch.testing.assert_close(v,old[k],atol=0,rtol=0)
        self.assertGreater(float(m.fusion.projection.weight.abs().sum()),0.)
        torch.testing.assert_close(m(x,mode="ts_only"),ts.numeric_forecast(x),atol=0,rtol=0)

    def test_empty_window_has_zero_correction_and_bound(self):
        m=self.model(aggregation_kind="count_mean",window_days=1e-6,gated=True).eval().requires_grad_(False)
        s=FrozenForecastSession(m,self.data.corpus)
        x,_,t=self.windows[0]
        r=s.predict(x[None],datetime_utc(t)+timedelta(minutes=1),.1)
        self.assertEqual(r.aggregation_semantics["visible_documents"],0)
        self.assertEqual(r.certified_bound,0.)
        np.testing.assert_allclose(r.prediction,m.numeric_forecast(torch.tensor(x[None])).numpy(),atol=0,rtol=0)

    def test_numeric_quarantine_removes_entire_conflicting_timestamp(self):
        from scripts.prepare_citra_weather import hourly_asos
        path=Path(self.tmp.name)/"raw.csv"
        payload="station,valid,tmpf,sknt\nDSM,2023-01-01 00:50,40,2\nDSM,2023-01-01 01:50,41,3\nDSM,2023-01-01 01:50,42,4\nDSM,2023-01-01 02:50,43,5\n"
        path.write_text(payload)
        (Path(str(path)+".metadata.json")).write_text(json.dumps(dict(sha256=hashlib.sha256(payload.encode()).hexdigest(),timestamp_timezone="UTC",station_id="DSM")))
        _,values,_=hourly_asos(path,"DSM",simulated_delay=60,excluded_timestamps={("DSM","2023-01-01 01:50")})
        self.assertTrue(np.isnan(values[1]).all())
        self.assertEqual(values[0,0],40.)
        self.assertEqual(values[2,0],43.)
        self.assertEqual(path.read_text(),payload)


if __name__=="__main__":unittest.main()
