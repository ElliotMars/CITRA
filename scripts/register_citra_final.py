"""Freeze the final protocol and provenance before any new outcome is evaluated."""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from citra.final_diagnosis import protocol_config_path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=="__main__":
    cfg=yaml.safe_load(protocol_config_path().read_text())
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=False)
    config_hash=digest(protocol_config_path())
    manifest=json.loads(Path(cfg["seen_data"],"manifest.json").read_text())
    sources={file:dict(expected=sha,actual=digest(file)) for file,sha in manifest["source_files"].items()}
    assert all(v["expected"]==v["actual"] for v in sources.values())
    checkpoints={}
    for seed in cfg["seeds"]:
        for name in ("window7_mean_residual","window7_mean_residual_shuffled"):
            file=Path(cfg["reference_root"])/name/f"seed_{seed}"/"full_asof_text.pt"
            checkpoints[str(file)]=digest(file)
        file=Path(cfg["ts_root"])/f"seed_{seed}"/"ts_only/ts_only.pt"
        checkpoints[str(file)]=digest(file)
    report=dict(config=cfg,config_sha256=config_hash,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        git_branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
        prompt_sha256=digest("prompt.md"),raw_sources=sources,checkpoints=checkpoints,
        text_encoder=manifest["frozen_text_encoder"],
        environment=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            platform=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)),
            preload=os.environ.get("LD_PRELOAD"),threads=dict(OPENBLAS_NUM_THREADS=os.getenv("OPENBLAS_NUM_THREADS"),OMP_NUM_THREADS=os.getenv("OMP_NUM_THREADS"))),
        roles={"2020":"train","2021":"validation","2022":"seen diagnostic","2023":"seen diagnostic","2024":"new only after this protocol lock"},
        decision_rule="System requires real seven-day output-budget benefit after full costs; synthetic alone cannot justify Go. Prediction requires content advantage over non-content and shuffle controls, not just total residual gain.",
        prior_capabilities="reuse AggregationSpec, independent exact_oracle, Temporal AVL, fit, frozen sessions, causal shuffle, append-only features; no new model/index",
        uncertainty="issuance_asof_assumption is not verified historical arrival; paired CIs concern dates in one region/year, not geographic generalization")
    (out/"registration.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"failures.jsonl").write_text("")
    print(json.dumps(dict(output=str(out),commit=report["git_commit"],config_sha256=config_hash),indent=2))
