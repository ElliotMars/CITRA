"""Synthetic scale checks; no duplicated-vector or real-workload claims."""
import argparse,csv,json,resource,subprocess,sys
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from citra import Document,TemporalDocumentStore
from citra.kernel import exact_aggregate
from citra.final_diagnosis import describe,protocol_config_path


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def prepare(cfg,out):
    import torch
    from scripts.run_citra_failure_ablation import load_data
    from citra.learning.training import load_checkpoint
    from citra.learning.inference import FrozenForecastSession
    torch.set_num_threads(1)
    data=load_data(cfg['train_data']);seed=cfg['system']['seed']
    checkpoint=Path(cfg['reference_root'])/'window7_mean_residual'/f'seed_{seed}'/'full_asof_text.pt'
    model,_=load_checkpoint(checkpoint,data,'cpu');model.eval().requires_grad_(False)
    session=FrozenForecastSession(model,data.corpus,build_index=False)
    keep=np.array([d.issued_at.year==2020 for d in data.corpus.documents])
    sample=np.concatenate([session.keys_np[keep],session.impacts_np[keep]],axis=1)
    covariance=np.cov(sample,rowvar=False);covariance+=np.eye(16)*(np.trace(covariance)/16*1e-5+1e-12)
    windows=data.windows('validation',48,6,[0,1],24);queries=[]
    for i in np.linspace(0,len(windows)-1,cfg['scaling']['query_count'],dtype=int):
        x,_,_=windows[int(i)]
        with torch.no_grad():queries.append(model.encode_query(torch.tensor(x[None],dtype=torch.float64)).numpy()[0])
    np.savez_compressed(out/'basis.npz',mean=sample.mean(0),covariance=covariance,queries=np.stack(queries),
        fusion=model.fusion.projection.weight.detach().numpy())
    write(out/'basis.json',dict(source='frozen selected seed7 checkpoint, 2020 document representations only',
        checkpoint=str(checkpoint),training_documents=int(keep.sum()),query_year=2021,
        distribution='joint Gaussian in key/impact space; keys re-normalized; jittered covariance; no copied real or duplicate vectors',
        limitations='Gaussian moments do not recreate semantic geometry, rare events, archive delays or real revision processes'))


def worker(cfg,out,size,scenario):
    begin=perf_counter();basis=np.load(out/'basis.npz');rng=np.random.default_rng(cfg['scaling']['seed']+size)
    t=datetime(2021,7,1,tzinfo=timezone.utc)
    extras=size if scenario=='mixed_archive' else 0
    revisions=int(size*cfg['scaling']['revision_fraction']) if scenario=='mixed_archive' else 0
    count=size+extras+revisions
    vectors=rng.multivariate_normal(basis['mean'],basis['covariance'],size=count,method='cholesky')
    z=vectors[:,:8];z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);v=vectors[:,8:]
    store=TemporalDocumentStore(8,8,sigma=.7,allow_simulated=True,indexed=False)
    cursor=0
    def document(doc_id,version,available):
        nonlocal cursor
        d=Document(doc_id,version,'synthetic-joint-Gaussian','synthetic',available,available,available,
            available+timedelta(days=7),None,None,z[cursor],v[cursor],availability_provenance='simulated')
        cursor+=1;return d
    for i in range(revisions):store.insert_document(document(f'active-{i}','old',t-timedelta(days=14+float(rng.random()))))
    for i in range(size):
        d=document(f'active-{i}','current',t-timedelta(days=float(rng.uniform(.001,6.99))))
        (store.append_revision if i<revisions else store.insert_document)(d)
    for i in range(extras):
        available=t-timedelta(days=float(rng.uniform(8,28))) if i<extras//2 else t+timedelta(days=float(rng.uniform(.01,7)))
        store.insert_document(document(f'other-{i}','v1',available))
    assert cursor==count
    data_seconds=perf_counter()-begin;s=perf_counter();store.build_index();construction=perf_counter()-s
    entries=tuple(e for node in store.index.nodes for e in node.document_refs)
    keys=np.stack([e.document.embedding for e in entries]);impacts=np.stack([e.document.impact_vector for e in entries])
    starts=np.array([e.start.timestamp() for e in entries]);ends=np.array([e.end.timestamp() if e.end else np.inf for e in entries])
    visible=store.snapshot(t);assert len(visible)==size
    # Exact fsum with an independently selected snapshot, outside timed calls.
    W=basis['fusion'];L=float(np.linalg.norm(W));queries=basis['queries'];oracle=[]
    for q in queries:oracle.append(W@(exact_aggregate(visible,q,store.kernel,8)/size))
    records=[]
    methods=[('numpy_exact',None)]+[('citra_avl',float(e)) for e in cfg['scaling']['epsilons']]
    def execute(q,j,method,epsilon,phase,repeat):
        total=perf_counter();s=perf_counter();mask=(starts<=t.timestamp())&(t.timestamp()<ends);n=int(mask.sum());assert n==size
        filtering=perf_counter()-s;den=max(n,1);stats=None;bound=0.
        s=perf_counter()
        if method=='numpy_exact':
            weights=np.exp(-((q-keys[mask])**2).sum(-1)/(2*.7**2));a=(weights[:,None]*impacts[mask]).sum(0)/den
        else:
            budget=0. if epsilon==0 else float(np.nextafter(epsilon/L*den,0.))
            r=store.query_certified(q,t,budget);a=r.aggregate/den;bound=L*r.aggregate_bound/den;stats=vars(r.stats)
        aggregation=perf_counter()-s;s=perf_counter();prediction=W@a;correction=perf_counter()-s;elapsed=perf_counter()-total
        deviation=float(np.linalg.norm(prediction-oracle[j]));assert deviation<=bound+1e-9 and (epsilon is None or bound<=epsilon)
        return dict(synthetic=True,scenario=scenario,legal_n=size,physical_records=count,method=method,epsilon=epsilon,
            phase=phase,repeat=repeat,query=j,filter_seconds=filtering,aggregation_seconds=aggregation,
            correction_seconds=correction,total_seconds=elapsed,actual_output_deviation=deviation,certified_bound=bound,
            scan_fraction=stats['documents_scanned']/size if stats else 1.,stats=stats)
    for method,e in methods:records.append(execute(queries[0],0,method,e,'cold_invocation',0))
    for repeat in range(cfg['scaling']['repeats']):
        for j,q in enumerate(queries):
            shift=(repeat+j)%len(methods)
            for method,e in methods[shift:]+methods[:shift]:records.append(execute(q,j,method,e,'warm',repeat))
    memory=dict(index_resident_bytes=store.index.resident_bytes,process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        encoded_array_bytes=keys.nbytes+impacts.nbytes,temporal_array_bytes=starts.nbytes+ends.nbytes)
    # Actual local append maintenance, outside the query benchmark. All affect
    # future cutoffs; benchmark history is not modified while queries run.
    update_rows=[]
    for i in range(10):
        zv=rng.multivariate_normal(basis['mean'],basis['covariance']);key=zv[:8]/np.linalg.norm(zv[:8]);impact=zv[8:]
        a=t+timedelta(minutes=i+1)
        d=Document(f'update-{i}','v1','synthetic-joint-Gaussian','synthetic',a,a,a,a+timedelta(days=7),None,None,key,impact,availability_provenance='simulated')
        s=perf_counter();store.insert_document(d);update_rows.append(dict(seconds=perf_counter()-s,**store.last_update_stats))
    summary=[]
    for method,e in methods:
        rs=[r for r in records if r['method']==method and r['epsilon']==e and r['phase']=='warm']
        summary.append(dict(method=method,epsilon=e,total=describe([r['total_seconds'] for r in rs]),
            aggregation=describe([r['aggregation_seconds'] for r in rs]),scan=describe([r['scan_fraction'] for r in rs])))
    result=dict(synthetic=True,legal_n=size,scenario=scenario,physical_records=count,old_revisions=revisions,
        expired_records=extras//2,future_records=extras-extras//2,data_preparation_seconds=data_seconds,
        construction_seconds=construction,memory=memory,maintenance=update_rows,records=records,summary=summary,
        elapsed_seconds=perf_counter()-begin,
        timing_scope='temporal mask + aggregation + frozen actual linear correction; query vectors precomputed, excludes history/neural query encoding',
        limitation='controlled random timing and 5% revisions are not a representative forecast archive; active-only is an optimistic homogeneous-time case')
    write(out/f'{scenario}_{size}.json',result)
    print(scenario,size,'construction_s',construction,'peak_MB',memory['process_peak_rss']/1e6,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','all','worker']);p.add_argument('--size',type=int);p.add_argument('--scenario',choices=['active_only','mixed_archive']);a=p.parse_args()
    cfg=yaml.safe_load(protocol_config_path().read_text());root=Path(cfg['output']);out=root/'scaling';out.mkdir(exist_ok=True)
    assert json.loads((root/'registration.json').read_text())['config']==json.loads(json.dumps(cfg))
    if a.stage=='prepare':prepare(cfg,out)
    elif a.stage=='worker':worker(cfg,out,a.size,a.scenario)
    else:
        for size in cfg['scaling']['sizes']:
            for scenario in cfg['scaling']['scenarios']:
                if (out/f'{scenario}_{size}.json').exists():continue
                with (out/f'{scenario}_{size}.log').open('w') as f:
                    process=subprocess.run([sys.executable,__file__,'worker','--size',str(size),'--scenario',scenario],stdout=f,stderr=subprocess.STDOUT)
                if process.returncode:
                    with (root/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(stage='scaling',size=size,scenario=scenario,returncode=process.returncode))+'\n')
