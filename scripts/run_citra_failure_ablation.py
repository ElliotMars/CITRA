"""Validation-only design selection, then one locked text operator on 2023."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_provider.citra_weather import WeatherData,datetime_utc
from citra.learning.training import seed_everything,load_checkpoint,fit,batch_tensors
from citra.learning.aggregation import AggregationSpec,exact_oracle
from citra.learning.inference import FrozenForecastSession,parameter_digest
from exp.exp_CITRA import error_metrics
from models.CITRA import Model


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def load_data(path):
    return WeatherData(path,availability_protocol="issuance_asof_assumption",text_delay_hours=0.)


@torch.no_grad()
def evaluate(model,data,split,output,year=None,text_disabled=False):
    model.eval()
    device=next(model.parameters()).device;c=model.config
    windows=data.windows(split,c["seq_len"],c["horizon"],c["target_ids"],24)
    indices=[i for i in range(len(windows)) if year is None or datetime_utc(windows[i][2]).year==year]
    if not indices:raise ValueError("no evaluation windows")
    spec=AggregationSpec.from_model(model)
    if not text_disabled:
        base,meta=data.corpus.tensors(device,torch.float64)
        keys,impacts=model.encode_documents(base,meta)
    pred,truth,numeric,gates,counts,cutoffs=[],[],[],[],[],[]
    for begin in range(0,len(indices),32):
        ix=indices[begin:begin+32]
        x,y,mask=batch_tensors(windows,ix,device,include_text=not text_disabled,aggregation_spec=spec)
        num=model.numeric_forecast(x)
        if text_disabled:
            cor=torch.zeros_like(num)
        else:
            q=model.encode_query(x);a=model.aggregate(q,keys,impacts,mask)
            cor=model.text_correction(x,a)
        pred.append((num+cor).cpu().numpy());numeric.append(num.cpu().numpy());truth.append(y.cpu().numpy())
        gates.extend([0.]*len(ix) if text_disabled else model.gate_value(x).cpu().numpy().ravel().tolist())
        counts.extend([0]*len(ix) if text_disabled else mask.sum(-1).cpu().tolist())
        cutoffs.extend(windows[i][2] for i in ix)
    pred,truth,numeric=map(np.concatenate,(pred,truth,numeric))
    def metrics(p):
        m=error_metrics(p,truth,data,c["target_ids"])
        m["normalized_bias_by_target"]=(p-truth).mean((0,1)).tolist()
        return m
    result=dict(split=split,year=year,windows=len(indices),text_on=metrics(pred),text_off=metrics(numeric),
        gate_mean=float(np.mean(gates)),visible_count_mean=float(np.mean(counts)),
        cohort_sha256=hashlib.sha256(np.asarray(cutoffs,dtype=np.int64).tobytes()).hexdigest(),
        model_config=c,data_fingerprint=data.fingerprint,scaler=dict(mean=data.mean.tolist(),scale=data.scale.tolist()))
    if output:
        np.savez_compressed(str(output)+".npz",prediction=pred,numeric=numeric,truth=truth,cutoff=np.asarray(cutoffs),gates=gates,counts=counts)
        write(str(output)+".json",result)
    return result


def train_variant(cfg,variant,seed,data,root,device):
    out=root/variant["name"]/f"seed_{seed}"
    if (out/"validation.json").exists():
        print("resume",out,flush=True);return
    out.mkdir(parents=True,exist_ok=False)
    data=data.with_causal_shuffle(seed) if variant.get("shuffled") else data
    config=dict(cfg,seed=seed)
    seed_everything(seed)
    if variant.get("reuse_original"):
        original=Path(cfg["ts_root"])/f"seed_{seed}"/"full_asof_text"
        model,state=load_checkpoint(original/"full_asof_text.pt",data,device)
        training=json.loads((original/"results.json").read_text())["models"]["full_asof_text"]["training"]
        checkpoint=original/"full_asof_text.pt"
    else:
        model=Model(seq_len=48,channels=2,horizon=6,base_dim=data.manifest["base_dim"],metadata_dim=2,
                    aggregation_kind=variant["kind"],window_days=variant.get("days"),gated=variant.get("gated",False)).double().to(device)
        if variant.get("residual"):
            ts,_=load_checkpoint(Path(cfg["ts_root"])/f"seed_{seed}"/"ts_only/ts_only.pt",data,device)
            model.backbone.load_state_dict(ts.backbone.state_dict());model.backbone.requires_grad_(False)
            torch.nn.init.zeros_(model.fusion.projection.weight)
            for split in ("train","validation"):
                x,_,_=batch_tensors(data.windows(split,48,6,[0,1],24),[0],device)
                torch.testing.assert_close(model(x,mode="ts_only"),ts.numeric_forecast(x),atol=0,rtol=0)
        before={k:v.detach().clone() for k,v in model.backbone.state_dict().items()}
        model,training=fit(model,data,config,"full_asof_text",out)
        if variant.get("residual"):
            for k,v in model.backbone.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        checkpoint=out/"full_asof_text.pt"
    write(out/"run.json",dict(variant=variant,seed=seed,checkpoint=str(checkpoint),training=training,
        data_fingerprint=data.fingerprint,numeric_fingerprint=data.numeric_fingerprint,
        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),model_config=model.config))
    evaluate(model,data,"train",out/"train")
    result=evaluate(model,data,"validation",out/"validation")
    print(variant["name"],seed,"val",result["text_on"]["normalized_mse"],flush=True)


def select(cfg,root):
    candidates=[]
    for v in cfg["variants"]:
        vals=[json.loads((root/v["name"]/f"seed_{s}"/"validation.json").read_text())["text_on"]["normalized_mse"] for s in cfg["seeds"]]
        candidates.append(dict(variant=v["name"],values=vals,mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)),select=v.get("select",True)))
    selected=min([v for v in candidates if v["select"]],key=lambda v:(v["mean"],v["variant"]))["variant"]
    decision=dict(selected=selected,candidates=candidates,criterion=cfg["selection"],
        seed_order=cfg["seeds"],used_test_labels=False,holdout_year=cfg["holdout_year"],
        note="2022 already seen. No 2022/2023 metrics were computed by this ablation runner before selection.")
    target=root/"selection.json"
    if target.exists() and json.loads(target.read_text())!=decision:raise ValueError("selection is already locked")
    write(target,decision)
    return decision


def append_only(old,new):
    n=len(old.times);m=len(old.corpus.records)
    for name in ("times","values","available"):
        np.testing.assert_array_equal(getattr(old,name),getattr(new,name)[:n])
    np.testing.assert_array_equal(old.mean,new.mean);np.testing.assert_array_equal(old.scale,new.scale)
    assert old.train_end==new.train_end and old.validation_end==new.validation_end
    assert list(old.corpus.records)==list(new.corpus.records[:m])


def load_run(root,v,seed,data,device):
    working=data.with_causal_shuffle(seed) if v.get("shuffled") else data
    record=json.loads((root/v["name"]/f"seed_{seed}"/"run.json").read_text())
    model,_=load_checkpoint(record["checkpoint"],working,device)
    return model,working


def heldout(cfg,root,device):
    decision=json.loads((root/"selection.json").read_text())
    chosen=next(v for v in cfg["variants"] if v["name"]==decision["selected"])
    old=load_data(cfg["tail_data_path"] if chosen.get("tail") else cfg["data_path"])
    # Construct/train a matched content control BEFORE looking at holdout labels.
    control=dict(chosen,name="selected_matched_shuffle",shuffled=True,reuse_original=False)
    for candidate in cfg["variants"]:
        if candidate.get("shuffled") and all(candidate.get(k,default)==chosen.get(k,default) for k,default in
                (("kind","sum"),("days",None),("residual",False),("gated",False),("tail",False))):
            control=candidate
            break
    for seed in cfg["seeds"]:train_variant(cfg,control,seed,old,root,device)
    control_lock=dict(selected=chosen,control=control,trained_seeds=cfg["seeds"],selection_sha256=hashlib.sha256((root/"selection.json").read_bytes()).hexdigest())
    if not (root/"holdout_lock.json").exists():write(root/"holdout_lock.json",control_lock)
    else:assert json.loads((root/"holdout_lock.json").read_text())==control_lock
    extended=load_data(cfg["tail_holdout_data_path"] if chosen.get("tail") else cfg["holdout_data_path"])
    append_only(old,extended)
    result=[]
    for seed in cfg["seeds"]:
        for variant in (chosen,control):
            model,_=load_run(root,variant,seed,old,device)
            target=root/variant["name"]/f"seed_{seed}"/"holdout_2023"
            working=extended.with_causal_shuffle(seed) if variant.get("shuffled") else extended
            # Matching old prefixes also for causal control: future arrivals cannot change past donors.
            if variant.get("shuffled"):append_only(old.with_causal_shuffle(seed),working)
            report=evaluate(model,working,"test",target,2023)
            result.append(dict(variant=variant["name"],seed=seed,**report))
            if not variant.get("shuffled"):
                shuffled=evaluate(model,extended.with_causal_shuffle(seed),"test",
                    root/variant["name"]/f"seed_{seed}"/"holdout_fixed_weights_shuffle",2023)
                result.append(dict(variant="selected_fixed_weights_shuffle",seed=seed,**shuffled))
        ts,_=load_checkpoint(Path(cfg["ts_root"])/f"seed_{seed}"/"ts_only/ts_only.pt",old,device)
        result.append(dict(variant="fixed_ts_only",seed=seed,**evaluate(ts,extended,"test",root/f"ts_seed_{seed}_holdout",2023,text_disabled=True)))
    write(root/"holdout_results.json",dict(selection=decision,rows=result,
        evaluation_role="new year after a validation-only locked selection; same region and issuance assumptions, not multiple independent environments"))


def retrospective_and_certify(cfg,root,device):
    if not (root/"selection.json").exists():raise ValueError("lock validation choice first")
    reports=[];certificates=[];certified_specs=set()
    for v in cfg["variants"]:
        data=load_data(cfg["tail_data_path"] if v.get("tail") else cfg["data_path"])
        for seed in cfg["seeds"]:
            model,working=load_run(root,v,seed,data,device)
            target=root/v["name"]/f"seed_{seed}"/"diagnostic_2022"
            reports.append(dict(variant=v["name"],seed=seed,**evaluate(model,working,"test",target,2022)))
            key=(v["kind"],v.get("days"),v.get("gated",False))
            if seed!=cfg["seeds"][0] or key in certified_specs:continue
            certified_specs.add(key)
            model.eval().requires_grad_(False)
            started=perf_counter();session=FrozenForecastSession(model,working.corpus)
            construction=perf_counter()-started
            windows=working.windows("validation",48,6,[0,1],24)
            for i in (0,len(windows)//2,len(windows)-1):
                x,_,t=windows[i];cutoff=datetime_utc(t)
                tx=torch.tensor(x[None],dtype=torch.float64,device=device)
                q=model.encode_query(tx).detach().cpu().numpy()[0]
                a,n=exact_oracle(working.corpus,session.keys_np,session.impacts_np,q,cutoff,.7,session.spec)
                with torch.no_grad():ref=(model.numeric_forecast(tx)+model.text_correction(tx,torch.tensor(a[None],device=device))).cpu().numpy()
                for epsilon in (0.,.01,.1):
                    fast=session.predict(x[None],cutoff,epsilon)
                    exact=session.predict(x[None],cutoff,execution="full_asof_text")
                    np.testing.assert_allclose(exact.prediction,ref,atol=1e-10,rtol=1e-10)
                    error=float(np.linalg.norm(fast.prediction-ref))
                    assert error<=fast.certified_bound+1e-9 and fast.certified_bound<=epsilon
                    certificates.append(dict(variant=v["name"],seed=seed,epsilon=epsilon,error=error,bound=fast.certified_bound,
                        cutoff=t,visible=n,semantics=fast.aggregation_semantics,model_version=session.model_version,
                        setup_seconds=construction,index_construction_seconds=session.offline_timings["index_construction_seconds"],
                        stats=vars(fast.query_stats)))
    write(root/"diagnostic_results.json",reports);write(root/"certificate_regression.json",certificates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["train","heldout","diagnostic"])
    p.add_argument("--config",type=Path,default=Path("configs/citra_failure.yaml"))
    p.add_argument("--output",type=Path,default=Path("artifacts/citra_failure_diagnosis/experiments"))
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());torch.set_num_threads(cfg["threads"])
    root=a.output;root.mkdir(parents=True,exist_ok=True)
    registration=root/"registration.json"
    if registration.exists():assert json.loads(registration.read_text())["config"]==cfg
    else:write(registration,dict(config=cfg,base_commit="17a1d95",test_role="2022 already seen; selection uses only 2021",
        code_sha256={str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ("models/CITRA.py","citra/learning/training.py","citra/learning/inference.py",__file__)}))
    start=perf_counter()
    if a.stage=="train":
        cache={}
        for variant in cfg["variants"]:
            path=cfg["tail_data_path"] if variant.get("tail") else cfg["data_path"]
            if path not in cache:cache[path]=load_data(path)
            for seed in cfg["seeds"]:train_variant(cfg,variant,seed,cache[path],root,a.device)
        print(json.dumps(select(cfg,root),indent=2),flush=True)
    elif a.stage=="heldout":heldout(cfg,root,a.device)
    else:retrospective_and_certify(cfg,root,a.device)
    write(root/(a.stage+"_cost.json"),dict(wall_seconds=perf_counter()-start,device=a.device,stage=a.stage))


if __name__=="__main__":main()
