"""Recompute stored real metrics, cohorts and source hashes without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_provider.citra_weather import WeatherData,load_jsonl
from scripts.encode_citra_text import load_features


def verify(root=Path("artifacts/citra_real_dsm"),hash_root=Path("artifacts/citra_real_hash"),
           data_root=Path("dataset/citra_real_dsm")):
    root,hash_root,data_root=Path(root),Path(hash_root),Path(data_root)
    data=WeatherData(data_root/"prepared",availability_protocol="issuance_asof_assumption",text_delay_hours=0)
    assert not data.manifest["is_fixture"]
    windows=data.windows("test",48,6,[0,1],24)
    truth=np.stack([windows[i][1] for i in range(len(windows))])
    cutoff=np.array([windows[i][2] for i in range(len(windows))],dtype=np.int64)
    for file,expected in data.manifest["source_files"].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
    attached,_=load_features(data_root/"minilm",load_jsonl(data_root/"collection/documents.jsonl"))
    np.testing.assert_array_equal(np.asarray([r["embedding"] for r in attached]),data.corpus.embeddings)
    unique=set()
    rows_count=0
    summaries=[root/"linear_summary.json",root/"dlinear_summary.json",hash_root/"linear_summary.json"]
    for file in summaries:
        summary=json.loads(file.read_text())
        assert summary["complete"] and len(summary["rows"])==summary["expected_rows"]
        for group in summary["groups"]:
            assert group["seeds"]==[7,17,29]
            rows=[r for r in summary["rows"] if r["mode"]==group["mode"] and r["delay_hours"]==group["delay_hours"]]
            for key,values in group["metrics"].items():
                np.testing.assert_allclose(values["mean"],np.mean([r[key] for r in rows]),rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(values["std"],np.std([r[key] for r in rows],ddof=1),rtol=1e-12,atol=1e-12)
        for row in summary["rows"]:
            rows_count+=1
            path=Path(row["result_path"])
            if str(path) in unique:
                continue
            unique.add(str(path))
            result=json.loads(path.read_text())
            config=json.loads((path.parent/"resolved_config.json").read_text())
            assert not result["is_fixture"]
            assert config["numeric_fingerprint"]==data.numeric_fingerprint
            assert config["scaler"]==dict(mean=data.mean.tolist(),scale=data.scale.tolist())
            with np.load(path.parent/(row["mode"]+"_predictions.npz"),allow_pickle=False) as pred:
                np.testing.assert_array_equal(pred["cutoff"],cutoff)
                np.testing.assert_array_equal(pred["truth_normalized"],truth)
                error=pred["prediction_normalized"]-truth
            metrics=result["models"][row["mode"]]["evaluation"]["metrics"]
            physical=error*data.scale
            for key,value in dict(normalized_mse=np.mean(error**2),normalized_mae=np.mean(np.abs(error)),
                                  mse=np.mean(physical**2),mae=np.mean(np.abs(physical))).items():
                np.testing.assert_allclose([row[key],metrics[key]],value,rtol=1e-12,atol=1e-12)
    pairs=0
    shared=[]
    for directory in (root/"profiles_final",root/"profiles_cpu"):
        protocol=json.loads((directory/"protocol.json").read_text())
        for file,expected in protocol["source_sha256"].items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
        ts=json.loads((directory/"ts_only/results.json").read_text())["models"]["ts_only"]["evaluation"]
        assert ts["offline_timings"]["index_construction_seconds"]==0.
        assert ts["offline_timings"]["index_maintenance_seconds"]==0.
        assert all(t["query_encoding_seconds"]==0. for q in ts["queries"] for t in q["timings"])
        full=json.loads((directory/"full_and_citra/results.json").read_text())["models"]["citra_certified"]["evaluation"]
        shared.append(full["model_version"])
        for epsilon,queries in full["certified"].items():
            for q in queries:
                assert q["actual_output_deviation"]<=q["certified_bound"]+1e-9
                assert q["certified_bound"]<=float(epsilon)
                pairs+=1
    assert len(set(shared))==1
    return dict(verified=True,source_files_checked=len(data.manifest["source_files"]),
        offline_feature_records_checked=len(attached),forecast_windows=len(windows),
        summary_rows_checked=rows_count,unique_prediction_runs_checked=len(unique),
        certificate_pairs_checked=pairs,shared_cpu_gpu_full_model=shared[0],
        note="Artifact consistency verification, not verification of historical public availability.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/citra_real_dsm"))
    p.add_argument("--hash-root",type=Path,default=Path("artifacts/citra_real_hash"))
    p.add_argument("--data",type=Path,default=Path("dataset/citra_real_dsm"))
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    report=verify(a.root,a.hash_root,a.data)
    payload=json.dumps(report,indent=2)+"\n"
    if a.output:
        a.output.write_text(payload)
    print(payload)


if __name__=="__main__":
    main()
