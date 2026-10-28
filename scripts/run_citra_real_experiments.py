"""Three-seed, four-delay real-data runs; no fixture masquerading as real results."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exp.exp_CITRA import run_experiment


MODES=("ts_only","semantic_topk","full_asof_text","shuffled_text")


def job_path(root,backbone,delay,seed,mode):
    return Path(root)/backbone/f"delay_{delay:g}h"/f"seed_{seed}"/mode


def train(config,root,device,backbone,modes=MODES,planned_modes=None):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    registered=copy.deepcopy(config)
    registered["model"]["backbone_kind"]=backbone
    registration=dict(config=registered,backbone=backbone,modes=list(planned_modes or modes),
        note="Fixed year split; all seeds reported; each text delay retrains with the same budget. No test-label tuning.")
    registry=root/f"{backbone}_registration.json"
    if registry.exists() and json.loads(registry.read_text())!=registration:
        raise ValueError("existing registration differs; choose another output directory")
    registry.write_text(json.dumps(registration,indent=2)+"\n")
    for delay in config["delays"]:
        for seed in config["seeds"]:
            for mode in modes:
                # Numerical evidence, splits and features do not change with
                # text delay. Fit TS-only once per seed, then reuse explicitly.
                if mode=="ts_only" and delay!=config["delays"][0]:
                    continue
                target=job_path(root,backbone,delay,seed,mode)
                if (target/"results.json").exists():
                    existing=json.loads((target/"results.json").read_text())
                    if existing["is_fixture"] or mode not in existing["models"]:
                        raise ValueError("existing run is incomplete or not real")
                    print("resume complete",target,flush=True)
                    continue
                cfg=copy.deepcopy(config)
                cfg.update(seed=seed,text_delay_hours=delay,profile_queries=0)
                if mode=="ts_only":
                    cfg["data_path"]=config.get("numeric_data_path",config["data_path"])
                cfg["model"]["backbone_kind"]=backbone
                start=perf_counter()
                print("training",backbone,delay,seed,mode,flush=True)
                result=run_experiment(cfg,target,device,mode)
                if result["is_fixture"]:
                    raise ValueError("real experiment refused fixture data")
                score=result["models"][mode]["evaluation"]["metrics"]
                print("finished",mode,"normalized_mse",score["normalized_mse"],"seconds",perf_counter()-start,flush=True)


def summarize(config,root,backbone):
    registry=Path(root)/f"{backbone}_registration.json"
    modes=json.loads(registry.read_text())["modes"] if registry.exists() else list(MODES)
    rows=[]
    reference_cutoffs=None
    for delay in config["delays"]:
        for seed in config["seeds"]:
            for mode in modes:
                actual_delay=config["delays"][0] if mode=="ts_only" else delay
                path=job_path(root,backbone,actual_delay,seed,mode)
                if not (path/"results.json").exists():
                    continue
                result=json.loads((path/"results.json").read_text())
                entry=result["models"][mode]
                evaluation=entry["evaluation"]
                pred=np.load(path/f"{mode}_predictions.npz",allow_pickle=False)
                cutoffs=pred["cutoff"]
                if reference_cutoffs is None:
                    reference_cutoffs=cutoffs.copy()
                elif not np.array_equal(reference_cutoffs,cutoffs):
                    raise ValueError("forecast cohorts differ across methods/delays/seeds")
                metrics=evaluation["metrics"]
                row=dict(backbone=backbone,delay_hours=delay,seed=seed,mode=mode,
                    evaluated_windows=evaluation["evaluated_windows"],**{k:metrics[k] for k in (
                        "mse","mae","normalized_mse","normalized_mae")},
                    training_seconds=entry["training"]["training_seconds"],
                    best_epoch=entry["training"]["best_epoch"],result_path=str(path/"results.json"),
                    ts_only_reused_from_delay=actual_delay if mode=="ts_only" else None)
                for target,values in metrics["per_target"].items():
                    for name,value in values.items():
                        row[target+"_"+name]=value
                rows.append(row)
    groups=[]
    for delay in config["delays"]:
        for mode in modes:
            selected=[r for r in rows if r["delay_hours"]==delay and r["mode"]==mode]
            if selected:
                score_keys=[k for k in selected[0] if k in ("mse","mae","normalized_mse","normalized_mae")
                            or k.endswith("_mse") or k.endswith("_mae")]
                groups.append(dict(delay_hours=delay,mode=mode,seeds=[r["seed"] for r in selected],
                    count=len(selected),metrics={k:dict(mean=float(np.mean([r[k] for r in selected])),
                    std=float(np.std([r[k] for r in selected],ddof=1)) if len(selected)>1 else None)
                    for k in score_keys}))
    unique_runs={r["result_path"]:r for r in rows}
    result=dict(availability_protocol=config["availability_protocol"],backbone=backbone,
        historical_first_publication_verified=False,rows=rows,groups=groups,
        unique_trained_models=len(unique_runs),
        unique_training_seconds=sum(r["training_seconds"] for r in unique_runs.values()),
        modes=modes,expected_rows=len(config["delays"])*len(config["seeds"])*len(modes),
        complete=len(rows)==len(config["delays"])*len(config["seeds"])*len(modes),
        forecast_cohort_sha256=hashlib.sha256(reference_cutoffs.tobytes()).hexdigest() if reference_cutoffs is not None else None,
        note="std across seeds measures training randomness, not climate/geographic uncertainty; physical aggregate mixes units")
    root=Path(root)
    (root/f"{backbone}_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if rows:
        with (root/f"{backbone}_scores.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    return result


def profile(config,root,device,backbone,output=None):
    # Predeclared diagnostic: the first registered seed/delay, evenly spaced
    # forecast windows. It is not a best-score or far-distance selection.
    seed,delay=config["seeds"][0],config["delays"][0]
    checkpoint=job_path(Path(root),backbone,delay,seed,"full_asof_text")/"full_asof_text.pt"
    cfg=copy.deepcopy(config)
    cfg.update(seed=seed,text_delay_hours=delay)
    cfg["model"]["backbone_kind"]=backbone
    return run_experiment(cfg,Path(output) if output else Path(root)/(backbone+"_profile"),device,"citra_certified",checkpoint)


def profile_all(config,root,device,backbone,output):
    """One sequential process, no training; report every registered diagnostic."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    seed,delay=config["seeds"][0],config["delays"][0]
    for mode in ("ts_only","semantic_topk"):
        cfg=copy.deepcopy(config)
        cfg.update(seed=seed,text_delay_hours=delay)
        if mode=="ts_only":
            cfg["data_path"]=config.get("numeric_data_path",config["data_path"])
        cfg["model"]["backbone_kind"]=backbone
        checkpoint=job_path(root,backbone,delay,seed,mode)/(mode+".pt")
        run_experiment(cfg,output/mode,device,mode,checkpoint)
        print("profile completed",mode,flush=True)
    profile(config,root,device,backbone,output/"full_and_citra")
    source_files=("citra/learning/inference.py","citra/learning/diagnostics.py","exp/exp_CITRA.py",__file__)
    protocol=dict(seed=seed,text_delay_hours=delay,device=device,
        queries=config.get("profile_queries",12),repeats=config.get("profile_repeats",3),
        order=["ts_only","semantic_topk","full_asof_text","citra_certified"],
        same_process_sequential=True,training_in_this_process=False,
        source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
        note="Fixed evenly spaced queries; cold calls retained; no best-run selection. Small diagnostic sample, not production throughput.")
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["train","baseline","summarize","profile","profile_all"])
    p.add_argument("--config",type=Path,default=Path("configs/citra_real.yaml"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--backbone",choices=["linear","dlinear"],default="linear")
    p.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    p.add_argument("--profile-output",type=Path)
    args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if args.stage in ("train","baseline"):
        train(cfg,args.output,args.device,args.backbone,["ts_only"] if args.stage=="baseline" else args.modes,
              planned_modes=MODES if args.stage=="baseline" else None)
        summarize(cfg,args.output,args.backbone)
    elif args.stage=="profile":
        profile(cfg,args.output,args.device,args.backbone,args.profile_output)
    elif args.stage=="profile_all":
        if args.profile_output is None:
            p.error("profile_all requires a new --profile-output directory")
        profile_all(cfg,args.output,args.device,args.backbone,args.profile_output)
    else:
        print(json.dumps(summarize(cfg,args.output,args.backbone),indent=2))


if __name__=="__main__":
    main()
