"""Summarize all seeds, verify saved predictions, and plot validation curves."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml


def stats(values):
    return dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),values=list(map(float,values)))


def summarize(root,cfg):
    rows=[];groups=[];curves={};cohorts={}
    for variant in cfg["variants"]:
        name=variant["name"];curves[name]=[]
        for seed in cfg["seeds"]:
            folder=root/name/f"seed_{seed}"
            run=json.loads((folder/"run.json").read_text());curves[name].append(run["training"]["epochs"])
            for split,file in (("train","train"),("validation","validation"),("diagnostic_2022","diagnostic_2022")):
                report=json.loads((folder/(file+".json")).read_text())
                with np.load(folder/(file+".npz")) as a:
                    error=a["prediction"]-a["truth"]
                    np.testing.assert_allclose(np.mean(error**2),report["text_on"]["normalized_mse"],rtol=1e-12)
                    if split not in cohorts:cohorts[split]=a["cutoff"].copy()
                    np.testing.assert_array_equal(cohorts[split],a["cutoff"])
                rows.append(dict(report,variant=name,seed=seed,split=split))
    heldout=json.loads((root/"holdout_results.json").read_text())
    for row in heldout["rows"]:rows.append(dict(row,split="holdout_2023"))
    for split in ("train","validation","diagnostic_2022","holdout_2023"):
        for name in sorted({r["variant"] for r in rows if r["split"]==split}):
            selected=[r for r in rows if r["split"]==split and r["variant"]==name]
            assert [r["seed"] for r in selected]==cfg["seeds"]
            metrics={}
            for branch in ("text_on","text_off"):
                metrics[branch]={key:stats([r[branch][key] for r in selected]) for key in ("normalized_mse","normalized_mae")}
                metrics[branch]["per_target"]={target:{metric:stats([r[branch]["per_target"][target][metric] for r in selected])
                    for metric in ("mse","mae")} for target in selected[0][branch]["per_target"]}
                metrics[branch]["normalized_bias_by_target"]=[stats([r[branch]["normalized_bias_by_target"][i] for r in selected]) for i in (0,1)]
            groups.append(dict(variant=name,split=split,seeds=cfg["seeds"],metrics=metrics,windows=selected[0]["windows"]))
    paired=[]
    chosen=heldout["selection"]["selected"]
    for seed in cfg["seeds"]:
        folder=root/chosen/f"seed_{seed}"
        with np.load(folder/"holdout_2023.npz") as real,np.load(root/f"ts_seed_{seed}_holdout.npz") as ts,np.load(folder/"holdout_fixed_weights_shuffle.npz") as shuffled:
            np.testing.assert_array_equal(real["truth"],ts["truth"]);np.testing.assert_array_equal(real["cutoff"],ts["cutoff"])
            np.testing.assert_array_equal(real["numeric"],ts["prediction"])
            e=(real["prediction"]-real["truth"])**2;t=(ts["prediction"]-ts["truth"])**2;s=(shuffled["prediction"]-shuffled["truth"])**2
            paired.append(dict(seed=seed,source_ts_equals_text_off=True,mse_gain_over_ts=float(np.mean(t-e)),
                mse_gain_over_fixed_weights_shuffle=float(np.mean(s-e)),
                daily_win_fraction_over_ts=float(np.mean(e.mean((1,2))<t.mean((1,2))))))
    result=dict(groups=groups,rows=rows,paired_holdout=paired,selection=heldout["selection"],
        reporting="all 3 seeds; normalized joint errors and physical per-target errors; 2022 already seen")
    (root/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (root/"scores.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["variant","seed","split","windows","normalized_mse","normalized_mae"])
        writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in ("variant","seed","split","windows")}|{k:r["text_on"][k] for k in ("normalized_mse","normalized_mae")})
    return result,curves


def plots(root,result,curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(16,10),constrained_layout=True)
    for ax,(name,runs) in zip(axes.flat,curves.items()):
        for field,color in (("training_loss","C0"),("validation_loss","C1")):
            array=np.asarray([[r[field] for r in rows] for rows in runs])
            mean,std=array.mean(0),array.std(0,ddof=1);x=np.arange(1,len(mean)+1)
            ax.plot(x,mean,color=color,label=field);ax.fill_between(x,mean-std,mean+std,color=color,alpha=.2)
        ax.set(title=name,xlabel="Epoch",ylabel="Normalized MSE");ax.legend(fontsize=7);ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    fig.savefig(root/"train_validation_curves.png",dpi=160);fig.savefig(root/"train_validation_curves.pdf");plt.close(fig)
    selected=result["selection"]["selected"]
    groups=[r for r in result["groups"] if r["split"]=="holdout_2023"]
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for i,g in enumerate(groups):
        m=g["metrics"]["text_on"]["normalized_mse"]
        ax.bar(i,m["mean"],yerr=m["std"],capsize=5,alpha=.6)
        ax.scatter([i]*3,m["values"],c="black",s=22,zorder=3)
    ax.set(xticks=range(len(groups)),xticklabels=[g["variant"].replace("window7_mean_residual","selected") for g in groups],
        ylabel="2023 normalized MSE (mean, seed std, all seeds)",title="Locked validation choice; same 364 daily forecast origins")
    ax.tick_params(axis="x",rotation=15);ax.grid(axis="y",alpha=.2)
    fig.savefig(root/"heldout_controls.png",dpi=160);fig.savefig(root/"heldout_controls.pdf");plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/citra_failure_diagnosis/experiments"))
    p.add_argument("--config",type=Path,default=Path("configs/citra_failure.yaml"))
    a=p.parse_args();result,curves=summarize(a.root,yaml.safe_load(a.config.read_text()));plots(a.root,result,curves)
