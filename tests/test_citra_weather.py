"""Offline parser fixtures: no real-data metric claims or network dependency."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from http.client import IncompleteRead
from unittest.mock import patch, MagicMock
import numpy as np

from data_provider.citra_weather import (UNKNOWN_TIME, WeatherData, make_fixture,
    parse_time, epoch_seconds, asof_forecast_covariates)
from scripts.prepare_citra_weather import afd_records, hourly_asos
from scripts.validate_citra_timestamps import audit
from scripts.citra_download import read_url


class WeatherAdapterTests(unittest.TestCase):
    def test_partial_http_response_is_retried(self):
        context = MagicMock()
        context.__enter__.return_value.read.side_effect = [IncompleteRead(b"partial"), b"complete"]
        with patch("scripts.citra_download.urlopen", return_value=context) as call, patch("scripts.citra_download.time.sleep"):
            self.assertEqual(read_url("https://fixture.invalid/data"), b"complete")
            self.assertEqual(call.call_count, 2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="citra-weather-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def metadata(self, path, **extra):
        row = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), retrieved_at="2026-10-28T00:00Z",
                   source_url="fixture://archive", timestamp_timezone="UTC", station_id="DSM")
        row.update(extra)
        Path(str(path) + ".metadata.json").write_text(json.dumps(row))

    def test_afd_issuance_is_not_availability(self):
        path = self.root / "afd.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("AFDDMX_202001010600.txt", "FXUS63 KDMX 010600\nAFDDMX\nOriginal fixture discussion")
            archive.writestr("202001010600-KDMX-FXUS63-AFDDMX-CCA.txt", "FXUS63 KDMX 010600 CCA\nAFDDMX\nCorrected fixture discussion")
        self.metadata(path)
        rows = afd_records(path, "DMX", "DSM")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["doc_id"], rows[1]["doc_id"])
        self.assertNotEqual(rows[0]["version_id"], rows[1]["version_id"])
        self.assertTrue(all(row["available_at"] is None and row["ingested_at"] is None for row in rows))
        simulated = afd_records(path, "DMX", "DSM", simulated_delay=20)
        for row in simulated:
            self.assertEqual(row["availability_provenance"], "simulated")
            self.assertEqual(parse_time(row["available_at"]) - parse_time(row["issued_at"]), timedelta(minutes=20))
        ledger = {rows[0]["sha256"]: dict(available_at="2020-01-01T06:20Z", evidence_url="fixture://evidence")}
        verified = afd_records(path, "DMX", "DSM", ledger=ledger)
        self.assertEqual(sum(r["availability_provenance"] == "verified" for r in verified), 1)

    def test_hour_aggregation_and_unknown_availability(self):
        path = self.root / "asos.csv"
        path.write_text("# fixture only\nstation,valid,tmpf,sknt\nDSM,2020-01-01 00:15,10,2\nDSM,2020-01-01 00:45,14,4\nDSM,2020-01-01 02:15,20,M\n")
        self.metadata(path)
        times, values, available = hourly_asos(path, "DSM")
        self.assertEqual(times[0], epoch_seconds(parse_time("2020-01-01T01:00Z")))
        np.testing.assert_array_equal(values[0], [12, 3])
        self.assertTrue(np.isnan(values[1]).all())
        self.assertTrue((available == UNKNOWN_TIME).all())
        _, _, delayed = hourly_asos(path, "DSM", simulated_delay=30)
        self.assertEqual(delayed[0, 0], epoch_seconds(parse_time("2020-01-01T01:15Z")))
        self.assertEqual(delayed[0, 1], delayed[0, 0])

    def test_fractional_availability_cannot_round_into_cutoff(self):
        path = self.root / "fractional.csv"
        row = dict(station="DSM", valid="2020-01-01 00:45", tmpf="10", sknt="2")
        path.write_text("station,valid,tmpf,sknt\nDSM,2020-01-01 00:45,10,2\n")
        self.metadata(path)
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        ledger = {key: dict(available_at="2020-01-01T01:00:00.000001Z", evidence_url="fixture://receipt")}
        times, _, available = hourly_asos(path, "DSM", ledger=ledger)
        self.assertGreater(available[0, 0], times[0])

    def test_download_hash_prevents_silent_raw_replacement(self):
        path = self.root / "asos.csv"
        path.write_text("station,valid,tmpf,sknt\nDSM,2020-01-01 00:15,10,2\n")
        self.metadata(path)
        path.write_text(path.read_text().replace("10,2", "20,2"))
        with self.assertRaises(ValueError):
            hourly_asos(path, "DSM")

    def test_audit_fixture_never_declares_real_strict_readiness(self):
        make_fixture(self.root / "data", hours=240)
        strict = audit(self.root / "data")
        self.assertFalse(strict["strict_backtest_ready"])
        self.assertTrue(strict["errors"])
        scenario = audit(self.root / "data", allow_simulated=True, seq_len=12, horizon=3)
        self.assertTrue(scenario["scenario_ready"])
        self.assertFalse(scenario["strict_backtest_ready"])

    def test_forecast_origins_respect_model_fitting_boundaries(self):
        make_fixture(self.root / "data", hours=240)
        data = WeatherData(self.root / "data", allow_simulated=True)
        for split, lower in (("validation", data.train_end), ("test", data.validation_end)):
            ds = data.windows(split, 12, 3, [0, 1])
            self.assertTrue(all(ds[i][2] >= lower for i in range(len(ds))))
        before = data.windows("validation", 12, 3, [0, 1])
        pos = before.starts[-1]
        # This truth is valid in the validation interval but was not available
        # by model-selection time. It must not select the test checkpoint.
        data.available[pos, 0] = data.validation_end + 3600
        after = data.windows("validation", 12, 3, [0, 1])
        self.assertNotIn(pos, after.starts)

    def test_asof_mos_extension_does_not_use_future_issuance(self):
        cutoff, valid = parse_time("2020-01-01T00:00Z"), parse_time("2020-01-01T12:00Z")
        base = dict(issued_at="2019-12-31T22:00Z", available_at="2019-12-31T22:05Z",
                    valid_time=valid.isoformat(), availability_provenance="verified", values={"temperature": 5.})
        future = dict(base, issued_at="2020-01-01T01:00Z", available_at="2020-01-01T01:05Z", values={"temperature": 999.})
        unknown = dict(base, available_at=None, values={"temperature": -999.})
        features = asof_forecast_covariates([base, future, unknown], cutoff, [valid], ["temperature", "wind"])
        self.assertEqual(features[0, 0], 5.)
        self.assertTrue(np.isnan(features[0, 1]))
