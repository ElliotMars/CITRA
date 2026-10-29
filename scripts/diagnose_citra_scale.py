"""Audit existing, unmodified Full checkpoints before designing interventions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_provider.citra_weather import WeatherData
from citra.learning.training import load_checkpoint,batch_tensors
from exp.exp_CITRA import error_metrics


def distribution(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),std=float(x.std()),p05=float(np.percentile(x,5)),
                p50=float(np.median(x)),p95=float(np.percentile(x,95)),min=float(x.min()),max=float(x.max()))


@torch.no_grad()
def diagnose(data,checkpoint,ts_checkpoint,device):
    model,state=load_checkpoint(checkpoint,data,device)
    ts,_=load_checkpoint(ts_checkpoint,data,device)
    model.eval().requires_grad_(False);ts.eval().requires_grad_(False)
    base,metadata=data.corpus.tensors(device,torch.float64)
    keys,impacts=model.encode_documents(base,metadata)
    c=model.config
    rows,groups=[],[]
    for split,year in (("train",2020),("validation",2021),("test",2022)):
        windows=data.windows(split,c["seq_len"],c["horizon"],c["target_ids"],24)
        predictions,numericals,standalone,truths,queries=[],[],[],[],[]
        for start in range(0,len(windows),32):
            indices=range(start,min(start+32,len(windows)))
            x,y,mask=batch_tensors(windows,indices,device)
            q=model.encode_query(x)
            weights=torch.exp(-((q[:,None]-keys[None])**2).sum(-1)/(2*c["sigma"]**2))*mask
            aggregate=weights@impacts
            correction=model.fusion(aggregate)
            numerical=model.numeric_forecast(x)
            full=numerical+correction
            n=mask.sum(-1)
            stats=dict(visible_documents=n,kernel_weight_sum=weights.sum(-1),
                kernel_weight_mean=weights.sum(-1)/n.clamp(min=1),kernel_weight_max=weights.max(-1).values,
                aggregate_l2=aggregate.norm(dim=-1),correction_l2=correction.flatten(1).norm(dim=-1),
                numeric_l2=numerical.flatten(1).norm(dim=-1),
                correction_to_numeric=correction.flatten(1).norm(dim=-1)/numerical.flatten(1).norm(dim=-1).clamp(min=1e-12),
                aggregate_l2_per_document=aggregate.norm(dim=-1)/n.clamp(min=1),
                correction_l2_per_document=correction.flatten(1).norm(dim=-1)/n.clamp(min=1))
            arrays={k:v.cpu().numpy() for k,v in stats.items()}
            for j,i in enumerate(indices):
                rows.append(dict(split=split,year=year,cutoff=windows[i][2],**{k:float(v[j]) for k,v in arrays.items()}))
            predictions.append(full.cpu().numpy());numericals.append(numerical.cpu().numpy())
            standalone.append(ts.numeric_forecast(x).cpu().numpy());truths.append(y.cpu().numpy());queries.append(q.cpu().numpy())
        truth=np.concatenate(truths)
        split_rows=[r for r in rows if r["split"]==split]
        metrics={}
        for name,parts in (("full",predictions),("full_checkpoint_text_off",numericals),("independent_ts_only",standalone)):
            pred=np.concatenate(parts)
            entry=error_metrics(pred,truth,data,c["target_ids"])
            entry["normalized_bias_by_target"]=(pred-truth).mean(axis=(0,1)).tolist()
            entry["physical_bias_by_target"]=(pred-truth).mean(axis=(0,1)).__mul__(data.scale[c["target_ids"]]).tolist()
            metrics[name]=entry
        groups.append(dict(split=split,year=year,windows=len(windows),metrics=metrics,
            distribution={k:distribution([r[k] for r in split_rows]) for k in stats},
            query_coordinate_mean=np.concatenate(queries).mean(0).tolist(),
            query_coordinate_std=np.concatenate(queries).std(0).tolist()))
    n=np.array([r["visible_documents"] for r in rows])
    correlations={k:float(np.corrcoef(n,[r[k] for r in rows])[0,1]) for k in
                  ("kernel_weight_sum","aggregate_l2","correction_l2","correction_to_numeric")}
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        model_config=c,best_epoch=state["best_epoch"],groups=groups,count_correlations=correlations,rows=rows)


def plots(reports,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    linear=[r for r in reports if r["backbone"]=="linear"]
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    metrics=("visible_documents","kernel_weight_sum","aggregate_l2","correction_l2","correction_to_numeric","aggregate_l2_per_document")
    for ax,key in zip(axes.flat,metrics):
        samples=[[row[key] for r in linear for row in r["rows"] if row["year"]==year] for year in (2020,2021,2022)]
        ax.boxplot(samples,tick_labels=["2020 train","2021 val","2022 diagnostic"],showfliers=False)
        ax.set_title(key);ax.grid(axis="y",alpha=.2)
    fig.savefig(output/"scale_distributions.png",dpi=160);fig.savefig(output/"scale_distributions.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for r in linear:
        axes[0].scatter([s["visible_documents"] for s in r["rows"]],[s["aggregate_l2"] for s in r["rows"]],s=3,alpha=.3,label=f'seed {r["seed"]}')
        for kind,style in (("full","-"),("full_checkpoint_text_off","--"),("independent_ts_only",":")):
            axes[1].plot([g["year"] for g in r["groups"]],[g["metrics"][kind]["normalized_mse"] for g in r["groups"]],style,label=f'{r["seed"]}: {kind}')
    axes[0].set(xlabel="Visible document count",ylabel="Aggregate L2");axes[0].legend()
    axes[1].set(xlabel="Year (2022 already seen)",ylabel="Normalized MSE",xticks=[2020,2021,2022]);axes[1].legend(fontsize=7)
    fig.savefig(output/"count_and_branch_drift.png",dpi=160);fig.savefig(output/"count_and_branch_drift.pdf");plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--data",default="dataset/citra_real_dsm/prepared")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);start=perf_counter()
    data=WeatherData(a.data,availability_protocol="issuance_asof_assumption",text_delay_hours=0.)
    reports=[]
    for backbone in ("linear","dlinear"):
        for seed in (7,17,29):
            root=Path("artifacts/citra_real_dsm")/backbone/"delay_0h"/f"seed_{seed}"
            result=diagnose(data,root/"full_asof_text/full_asof_text.pt",root/"ts_only/ts_only.pt",a.device)
            result.update(backbone=backbone,seed=seed);reports.append(result)
            print("audited",backbone,seed,flush=True)
    flat=[dict(backbone=r["backbone"],seed=r["seed"],**row) for r in reports for row in r["rows"]]
    with (a.output/"windows.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    payload=dict(reports=[{k:v for k,v in r.items() if k!="rows"} for r in reports],elapsed_seconds=perf_counter()-start,device=a.device,
        code_sha256={str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ("models/CITRA.py","citra/learning/training.py","citra/learning/inference.py")},
        interpretation="Frozen checkpoint diagnostics; training-year retrospective predictions use the final fitted model, not online out-of-sample predictions.",
        test_role="2022 is already observed and cannot be used as untouched final holdout.")
    (a.output/"diagnosis.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    plots(reports,a.output)


if __name__=="__main__":main()
