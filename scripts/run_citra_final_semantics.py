"""Fixed-capacity final content controls; only 2020/2021 train/select epochs."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_citra_failure_ablation import load_data,append_only,evaluate
from citra.final_diagnosis import control_view,paired_block_bootstrap,protocol_config_path
from citra.learning.training import load_checkpoint,seed_everything,fit
from citra.learning.inference import parameter_digest
from models.CITRA import Model


def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def transformed(data,kind,seed):
    if kind in ('shuffle_retrained','shuffle_fixed_weights'):return data.with_causal_shuffle(seed)
    if kind in ('real_text','ts_only'):return data
    return control_view(data,kind)


def train_controls(cfg,root,device):
    data=load_data(cfg['train_data'])
    for name in cfg['controls']:
        controlled=control_view(data,name)
        for seed in cfg['seeds']:
            folder=root/name/f'seed_{seed}'
            if (folder/'run.json').exists():continue
            folder.mkdir(parents=True,exist_ok=False);seed_everything(seed)
            model=Model(seq_len=48,channels=2,horizon=6,base_dim=384,metadata_dim=2,
                aggregation_kind='count_mean',window_days=7).double().to(device)
            ts_file=Path(cfg['ts_root'])/f'seed_{seed}'/'ts_only/ts_only.pt'
            ts,_=load_checkpoint(ts_file,data,device)
            model.backbone.load_state_dict(ts.backbone.state_dict());model.backbone.requires_grad_(False)
            torch.nn.init.zeros_(model.fusion.projection.weight)
            before={k:v.detach().clone() for k,v in model.backbone.state_dict().items()}
            model,training=fit(model,controlled,dict(cfg,seed=seed),'full_asof_text',folder)
            for k,v in model.backbone.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
            evaluate(model,controlled,'train',folder/'train')
            val=evaluate(model,controlled,'validation',folder/'validation')
            write(folder/'run.json',dict(kind=name,seed=seed,checkpoint=str(folder/'full_asof_text.pt'),training=training,
                total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
                model_config=model.config,backbone_source=str(ts_file),backbone_frozen_bit_exact=True,
                input_policy=cfg['controls'][name],data_fingerprint=controlled.fingerprint,
                input_capacity_note='allocated network identical; constant inputs reduce effective functional capacity and leave input-weight directions inactive'))
            print('trained',name,seed,'validation_mse',val['text_on']['normalized_mse'],flush=True)
    lock=dict(config_sha256=hashlib.sha256(protocol_config_path().read_bytes()).hexdigest(),
        new_checkpoints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*/seed_*/*.pt')},
        reused_checkpoints=json.loads((Path(cfg['output'])/'registration.json').read_text())['checkpoints'],
        no_new_model_selection='all methods evaluated; best epoch selected with 2021 only; no ranking/tuning with 2022/2023/2024',
        code_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('citra/final_diagnosis.py','models/CITRA.py','citra/learning/training.py',__file__)})
    target=root/'methods_frozen.json'
    if target.exists():assert json.loads(target.read_text())==lock
    else:write(target,lock)


def evaluate_methods(cfg,root,device,include_new):
    if not (root/'methods_frozen.json').exists():raise ValueError('freeze every method before evaluating outcomes')
    original=load_data(cfg['train_data'])
    data=load_data(cfg['new_data'] if include_new else cfg['seen_data']);append_only(original,data)
    all_names=['real_text','metadata_only','content_only','constant_text','null_no_metadata',
               'shuffle_retrained','shuffle_fixed_weights','ts_only']
    rows=[]
    # Fixed calendar years; all models, never the best seed or best test variant.
    years=[2022,2023]+([cfg['new_year']] if include_new else [])
    for name in all_names:
        for seed in cfg['seeds']:
            prior_working=transformed(original,name,seed);working=transformed(data,name,seed)
            append_only(prior_working,working)
            if name in cfg['controls']:
                checkpoint=root/name/f'seed_{seed}'/'full_asof_text.pt'
            elif name=='ts_only':checkpoint=Path(cfg['ts_root'])/f'seed_{seed}'/'ts_only/ts_only.pt'
            else:
                ref='window7_mean_residual_shuffled' if name=='shuffle_retrained' else 'window7_mean_residual'
                checkpoint=Path(cfg['reference_root'])/ref/f'seed_{seed}'/'full_asof_text.pt'
            # The fixed-weight intervention intentionally loads the REAL model
            # against its original corpus, then freshly encodes controlled input.
            load_against=original if name=='shuffle_fixed_weights' else prior_working
            model,_=load_checkpoint(checkpoint,load_against,device)
            folder=root/name/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if name not in cfg['controls']:
                for split in ('train','validation'):evaluate(model,working,split,folder/split,text_disabled=name=='ts_only')
            for year in years:
                result=evaluate(model,working,'test',folder/f'year_{year}',year,text_disabled=name=='ts_only')
                rows.append(dict(result,kind=name,seed=seed,checkpoint=str(checkpoint),year=year))
                print('evaluated',name,seed,year,result['text_on']['normalized_mse'],flush=True)
    write(root/'evaluations.json',dict(rows=rows,years=years,new_year_used=include_new,
        new_year_role='pre-registered once; all methods frozen before results' if include_new else 'no fresh independent year available',
        seen_years=[2022,2023],new_data_manifest=data.manifest))


def analyze(cfg,root):
    evaluations=json.loads((root/'evaluations.json').read_text());rows=evaluations['rows'];years=evaluations['years']
    groups=[];losses={};daily_rows=[]
    for name in sorted({r['kind'] for r in rows}):
        for year in years:
            selected=[r for r in rows if r['kind']==name and r['year']==year]
            assert [r['seed'] for r in selected]==cfg['seeds']
            metrics={k:dict(mean=float(np.mean([r['text_on'][k] for r in selected])),std=float(np.std([r['text_on'][k] for r in selected],ddof=1)),
                values=[r['text_on'][k] for r in selected]) for k in ('normalized_mse','normalized_mae')}
            metrics['per_target']={t:{metric:dict(mean=float(np.mean([r['text_on']['per_target'][t][metric] for r in selected])),
                std=float(np.std([r['text_on']['per_target'][t][metric] for r in selected],ddof=1))) for metric in ('mse','mae')} for t in selected[0]['text_on']['per_target']}
            seed_losses=[];dates=None
            for r in selected:
                f=root/name/f"seed_{r['seed']}"/f'year_{year}.npz'
                with np.load(f) as a:
                    current=a['cutoff'].astype('datetime64[s]').astype('datetime64[D]')
                    if dates is None:dates=current
                    np.testing.assert_array_equal(dates,current)
                    error=a['prediction']-a['truth'];scale=np.asarray(r['scaler']['scale'])
                    physical=error*scale;cor=a['prediction']-a['numeric']
                    ls=np.stack([np.mean(error**2,axis=(1,2)),np.mean(np.abs(error),axis=(1,2)),
                        np.mean(physical[:,:,0]**2,axis=1),np.mean(np.abs(physical[:,:,0]),axis=1),
                        np.mean(physical[:,:,1]**2,axis=1),np.mean(np.abs(physical[:,:,1]),axis=1)],axis=1)
                    seed_losses.append(ls)
                    for i,date in enumerate(dates):
                        daily_rows.append(dict(kind=name,year=year,seed=r['seed'],date=str(date),cutoff=int(a['cutoff'][i]),
                            normalized_mse=float(ls[i,0]),normalized_mae=float(ls[i,1]),temperature_mse=float(ls[i,2]),temperature_mae=float(ls[i,3]),
                            wind_mse=float(ls[i,4]),wind_mae=float(ls[i,5]),temperature_bias=float(physical[i,:,0].mean()),wind_bias=float(physical[i,:,1].mean()),
                            correction_l2=float(np.linalg.norm(cor[i])),numeric_l2=float(np.linalg.norm(a['numeric'][i]))))
                    if name!='ts_only':
                        ts=np.load(root/'ts_only'/f"seed_{r['seed']}"/f'year_{year}.npz')
                        np.testing.assert_array_equal(a['numeric'],ts['prediction'])
                        np.testing.assert_array_equal(a['truth'],ts['truth']);np.testing.assert_array_equal(a['cutoff'],ts['cutoff'])
            losses[name,year]=(dates,np.stack(seed_losses))
            relevant=[r for r in daily_rows if r['kind']==name and r['year']==year]
            groups.append(dict(kind=name,year=year,windows=selected[0]['windows'],metrics=metrics,
                correction_l2_mean=float(np.mean([r['correction_l2'] for r in relevant])),
                physical_bias_mean=[float(np.mean([r[key] for r in relevant])) for key in ('temperature_bias','wind_bias')]))
    paired=[];metric_names=('normalized_mse','normalized_mae','temperature_mse','temperature_mae','wind_mse','wind_mae')
    for year in years:
        dates,real=losses['real_text',year]
        for name in sorted({r['kind'] for r in rows}-{'real_text'}):
            other_dates,other=losses[name,year];np.testing.assert_array_equal(dates,other_dates)
            for j,metric in enumerate(metric_names):
                for block in cfg['bootstrap']['block_days']:
                    result=paired_block_bootstrap(dates,real[:,:,j]-other[:,:,j],block,cfg['bootstrap']['repeats'],cfg['bootstrap']['seed']+year)
                    paired.append(dict(year=year,contrast='real_text minus '+name,metric=metric,**result))
    write(root/'summary.json',dict(groups=groups,paired=paired,new_year_used=evaluations['new_year_used'],
        note='negative paired difference favors real content; seed-averaged calendar dates are resampled, not flattened seeds; CIs unadjusted for multiple comparisons'))
    write(root/'paired_bootstrap.json',paired)
    with (root/'daily_losses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(daily_rows[0]));w.writeheader();w.writerows(daily_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','evaluate','analyze']);p.add_argument('--device',default='cuda:0');p.add_argument('--include-new',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(protocol_config_path().read_text());base=Path(cfg['output'])
    assert json.loads((base/'registration.json').read_text())['config']==json.loads(json.dumps(cfg))
    root=base/'semantics';root.mkdir(exist_ok=True);torch.set_num_threads(1);s=perf_counter()
    if a.stage=='train':train_controls(cfg,root,a.device)
    elif a.stage=='evaluate':evaluate_methods(cfg,root,a.device,a.include_new)
    else:analyze(cfg,root)
    write(root/(a.stage+'_cost.json'),dict(wall_seconds=perf_counter()-s,device=a.device,stage=a.stage))


if __name__=='__main__':main()
