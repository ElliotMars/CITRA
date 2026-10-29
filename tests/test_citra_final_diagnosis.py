"""Content exclusion and date-level, serially dependent inference regressions."""
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
import numpy as np
from citra.final_diagnosis import control_view,SharedWindowMetadata,paired_block_bootstrap
from citra.learning.aggregation import AggregationSpec
from data_provider.citra_weather import make_fixture,WeatherData,parse_time


class FinalContentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        make_fixture(Path(self.tmp.name)/'data',hours=180,base_dim=8)
        self.data=WeatherData(Path(self.tmp.name)/'data',allow_simulated=True)

    def test_noncontent_controls_cannot_read_embedding_or_text(self):
        class Guarded(dict):
            def __getitem__(self,key):
                if key in ('embedding','text'):raise AssertionError('semantic input was accessed')
                return super().__getitem__(key)
        self.data.corpus.records=tuple(Guarded(r) for r in self.data.corpus.records)
        for kind in ('metadata_only','constant_text','null_no_metadata'):
            result=control_view(self.data,kind)
            self.assertEqual(result.corpus.embeddings.shape,self.data.corpus.embeddings.shape)
            expected=np.zeros_like(result.corpus.embeddings)
            if kind=='constant_text':expected[:,0]=1.
            np.testing.assert_array_equal(result.corpus.embeddings,expected)
            self.assertNotIn('frozen_text_encoder',result.manifest)

    def test_content_only_removes_direct_season_metadata(self):
        result=control_view(self.data,'content_only')
        np.testing.assert_array_equal(result.corpus.embeddings,self.data.corpus.embeddings)
        np.testing.assert_array_equal(result.corpus.metadata,np.zeros_like(result.corpus.metadata))
        self.assertFalse(np.array_equal(self.data.corpus.metadata,result.corpus.metadata))

    def test_controls_preserve_numerical_data_and_legal_sets(self):
        cuts=list(self.data.times[::9])
        for kind in ('metadata_only','constant_text','content_only','null_no_metadata'):
            result=control_view(self.data,kind)
            self.assertEqual(result.numeric_fingerprint,self.data.numeric_fingerprint)
            np.testing.assert_array_equal(result.mean,self.data.mean)
            np.testing.assert_array_equal(result.corpus.visibility(cuts),self.data.corpus.visibility(cuts))
            self.assertNotEqual(result.fingerprint,self.data.fingerprint)

    def test_shared_cached_filter_matches_version_and_window_oracles(self):
        corpus=self.data.corpus;cache=SharedWindowMetadata(corpus,.25);spec=AggregationSpec('count_mean',.25)
        origin=parse_time(corpus.records[0]['issued_at'])
        for delta in (-1,0,6,9,24,72,180):
            for micro in (0,1):
                cutoff=origin+timedelta(hours=delta,microseconds=micro)
                mask=cache.mask(cutoff)
                np.testing.assert_array_equal(mask,spec.mask(corpus,[cutoff])[0])
                expected={(d.doc_id,d.version_id) for d in corpus.store.snapshot(cutoff) if d.available_at>cutoff-timedelta(hours=6)}
                actual={(d.doc_id,d.version_id) for d,keep in zip(corpus.documents,mask) if keep}
                self.assertEqual(actual,expected)


class PairedDateInferenceTests(unittest.TestCase):
    def dates(self):return np.arange(np.datetime64('2023-01-01'),np.datetime64('2024-01-01'))

    def test_repeated_training_seeds_do_not_triple_weather_sample_size(self):
        dates=self.dates();difference=np.sin(np.arange(len(dates))/25)[None]*.01-.001
        one=paired_block_bootstrap(dates,difference,repeats=500)
        three=paired_block_bootstrap(dates,np.repeat(difference,3,axis=0),repeats=500)
        self.assertEqual(three['dates'],365)
        for key in ('mean_difference','ci95_low','ci95_high'):
            self.assertAlmostEqual(one[key],three[key],places=14)

    def test_constant_negative_difference_has_correct_direction(self):
        r=paired_block_bootstrap(self.dates(),np.full((3,365),-.002),repeats=100)
        self.assertAlmostEqual(r['ci95_high'],-.002)
        self.assertAlmostEqual(r['ci95_low'],-.002)
        self.assertEqual(r['improvement_date_fraction'],1.)

    def test_blocks_keep_serial_dependence_and_calendar_gaps(self):
        dates=self.dates();d=np.sin(np.arange(365)/35)[None]*.01
        iid=paired_block_bootstrap(dates,d,1,500)
        blocked=paired_block_bootstrap(dates,d,14,500)
        self.assertGreater(blocked['ci95_high']-blocked['ci95_low'],iid['ci95_high']-iid['ci95_low'])
        keep=np.ones(365,dtype=bool);keep[[2,40,100]]=False
        missing=paired_block_bootstrap(dates[keep],d[:,keep],7,100)
        self.assertEqual(missing['dates'],362)

    def test_duplicate_dates_and_cross_years_are_rejected(self):
        with self.assertRaises(ValueError):paired_block_bootstrap(['2023-01-01','2023-01-01'],np.ones((3,2)))
        with self.assertRaises(ValueError):paired_block_bootstrap(['2023-12-31','2024-01-01'],np.ones((3,2)))


if __name__=='__main__':unittest.main()
