"""Seven-day exact/AVL feasibility with identical frozen data and query semantics."""
import argparse,csv,json,math,resource,sys
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
from dataclasses import replace
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_citra_failure_ablation import load_data,append_only
from citra.learning.training import load_checkpoint
from citra.learning.inference import FrozenForecastSession
from citra.learning.aggregation import exact_oracle
from citra.learning.diagnostics import maintenance
from citra.final_diagnosis import SharedWindowMetadata,describe,protocol_config_path
from data_provider.citra_weather import datetime_utc,epoch_seconds
from citra import TemporalDocumentStore


def rss():
    return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-mode',choices=['full_retrospective','asof_prefix'],default='full_retrospective')
    args=parser.parse_args();prefix_mode=args.archive_mode=='asof_prefix'
    cfg=yaml.safe_load(protocol_config_path().read_text());root=Path(cfg['output'])
    assert json.loads((root/'registration.json').read_text())['config']==json.loads(json.dumps(cfg))
    out=root/('system_asof' if prefix_mode else 'system');out.mkdir(exist_ok=False)
    torch.set_num_threads(1);started=perf_counter();memory=dict(start_rss=rss())
    old=load_data(cfg['train_data']);data=load_data(cfg['seen_data']);append_only(old,data)
    seed=cfg['system']['seed'];checkpoint=Path(cfg['reference_root'])/'window7_mean_residual'/f'seed_{seed}'/'full_asof_text.pt'
    model,_=load_checkpoint(checkpoint,old,'cpu');model.eval().requires_grad_(False)
    assert model.config.get('window_days')==7 and model.config.get('aggregation_kind')=='count_mean'
    memory['data_and_model_rss']=rss()
    session=FrozenForecastSession(model,data.corpus,build_index=False)
    cached=SharedWindowMetadata(data.corpus,7)
    memory['representations_rss']=rss()
    rows=[];queries=[]
    docs=data.corpus.documents;maxint=np.iinfo(np.int64).max
    activate=np.array([max(epoch_seconds(t)*1000000+t.microsecond for t in (d.available_at,d.valid_from,d.ingested_at,d.issued_at) if t is not None)
        if d.available_at is not None and d.availability_provenance in ('verified','simulated') else maxint for d in docs],dtype=np.int64)
    ends=np.array([epoch_seconds(d.valid_to)*1000000+d.valid_to.microsecond if d.valid_to else maxint for d in docs],dtype=np.int64)
    # Count every actual daily forecast origin once, not once per training seed.
    for split in ('train','validation','test'):
        windows=data.windows(split,48,6,[0,1],24)
        by_year={}
        for i,pos in enumerate(windows.starts):
            t=int(data.times[pos-1]);cutoff=datetime_utc(t);year=cutoff.year
            full=data.corpus.visibility([t])[0];mask=cached.mask(t)
            np.testing.assert_array_equal(mask,session.spec.mask(data.corpus,[t])[0])
            eligible=(activate<=t*1000000)&(ends>t*1000000)
            raw_window=eligible&(cached.available>t*1000000-cached.duration_us)
            # Independent version oracle is checked on EVERY count query.
            oracle=data.corpus.store.snapshot(cutoff)
            oracle=[d for d in oracle if d.available_at>cutoff-__import__('datetime').timedelta(days=7)]
            assert len(oracle)==int(mask.sum())
            rows.append(dict(year=year,split=split,cutoff=t,date=cutoff.date().isoformat(),
                corpus_rows=len(docs),before_version_eligible=int(eligible.sum()),
                after_version_all_history=int(full.sum()),before_version_window_candidates=int(raw_window.sum()),
                legal_window_documents=int(mask.sum()),vector_metadata_rows=len(docs),exact_key_reads=int(mask.sum())))
            by_year.setdefault(year,[]).append((windows,i,t))
        for year,items in by_year.items():
            for i in np.unique(np.linspace(0,len(items)-1,cfg['system']['queries_per_year'],dtype=int)):
                queries.append(items[int(i)])
    with (out/'document_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    counts={str(year):dict(legal_documents=describe([r['legal_window_documents'] for r in rows if r['year']==year]),
        empty_ratio=float(np.mean([r['legal_window_documents']==0 for r in rows if r['year']==year])),
        before_version_window=describe([r['before_version_window_candidates'] for r in rows if r['year']==year]),
        full_history_after_versions=describe([r['after_version_all_history'] for r in rows if r['year']==year])) for year in sorted({r['year'] for r in rows})}
    write(out/'document_counts.json',dict(years=counts,all=describe([r['legal_window_documents'] for r in rows]),
        actual_origins=len(rows),metadata_cache_bytes=cached.bytes,
        note='retained source corpus; pre-preparation quarantines are in source manifests, not legal candidates; std is population std'))
    # All methods receive these SAME encoded CPU float64 arrays. CUDA only
    # offloads aggregation and includes both input and output transfers.
    cache_start=perf_counter()
    kcpu=torch.tensor(session.keys_np.copy());vcpu=torch.tensor(session.impacts_np.copy())
    kgpu=vgpu=None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats();kgpu=kcpu.to('cuda:0');vgpu=vcpu.to('cuda:0');torch.cuda.synchronize()
    cache_seconds=perf_counter()-cache_start
    if not prefix_mode:session.prepare_index()
    memory['with_index_and_caches_rss']=rss()
    index_bytes=session.store.index.resident_bytes if not prefix_mode else 0
    prefix_builds=[]
    active_store=session.store;active_k=session.keys_np;active_v=session.impacts_np
    active_starts=data.corpus._starts;active_ends=data.corpus._ends;active_available=cached.available

    def prepare_prefix(t):
        nonlocal active_store,active_k,active_v,active_starts,active_ends,active_available,kcpu,vcpu,kgpu,vgpu,index_bytes
        cutoff=datetime_utc(t);past=np.array([d.accessible(cutoff,data.corpus.allow_simulated) for d in docs])
        positions=np.flatnonzero(past);s=perf_counter()
        active_k=session.keys_np[positions];active_v=session.impacts_np[positions]
        active_starts=data.corpus._starts[positions];active_ends=data.corpus._ends[positions];active_available=cached.available[positions]
        active_store=TemporalDocumentStore(8,8,sigma=.7,allow_simulated=True,indexed=False);seen=set()
        for i in positions:
            d=session.spec.indexed_document(replace(docs[int(i)],embedding=session.keys_np[i],impact_vector=session.impacts_np[i]))
            if d is None:continue
            (active_store.append_revision if d.doc_id in seen else active_store.insert_document)(d);seen.add(d.doc_id)
        metadata_seconds=perf_counter()-s;s=perf_counter();active_store.build_index();build_seconds=perf_counter()-s
        s=perf_counter();kcpu=torch.tensor(active_k.copy());vcpu=torch.tensor(active_v.copy())
        if kgpu is not None:
            kgpu=kcpu.to('cuda:0');vgpu=vcpu.to('cuda:0');torch.cuda.synchronize()
        transfer_seconds=perf_counter()-s;resident=active_store.index.resident_bytes;index_bytes=max(index_bytes,resident)
        prefix_builds.append(dict(cutoff=t,past_records=len(positions),metadata_setup_seconds=metadata_seconds,
            index_build_seconds=build_seconds,representation_cache_seconds=transfer_seconds,index_resident_bytes=resident))
    methods=[('numpy_exact',None),('torch_cpu_exact',None)]
    if kgpu is not None:methods.append(('torch_cuda_exact',None))
    methods += [('citra_avl',float(e)) for e in cfg['system']['epsilons']]
    references={}
    for ds,i,t in queries:
        x=ds.history(ds.starts[i]);tx=torch.tensor(x[None],dtype=torch.float64)
        with torch.no_grad():q=model.encode_query(tx).numpy()[0]
        a,n=exact_oracle(data.corpus,session.keys_np,session.impacts_np,q,datetime_utc(t),.7,session.spec)
        with torch.no_grad():pred=(model.numeric_forecast(tx)+model.text_correction(tx,torch.tensor(a[None]))).numpy()
        references[t]=(a,pred,n)

    @torch.no_grad()
    def execute(ds,i,t,method,epsilon,phase,repeat):
        total=perf_counter();timings={};s=perf_counter();session.assert_current();timings['version_validation_seconds']=perf_counter()-s
        s=perf_counter();x=ds.history(ds.starts[i]);tx=torch.tensor(x[None],dtype=torch.float64);timings['history_preparation_seconds']=perf_counter()-s
        s=perf_counter();micros=t*1000000;mask=(micros>=active_starts)&(micros<active_ends)&(active_available>micros-cached.duration_us);positions=np.flatnonzero(mask);n=len(positions);den=max(n,1);timings['temporal_filter_seconds']=perf_counter()-s
        s=perf_counter();q=model.encode_query(tx).numpy()[0];timings['query_encoding_seconds']=perf_counter()-s
        stats=None;raw_bound=0.;aggregation_budget=0.
        timings.update(h2d_seconds=0.,d2h_seconds=0.,gather_conversion_seconds=0.)
        if method=='numpy_exact':
            s=perf_counter();z=active_k[positions];v=active_v[positions];timings['gather_conversion_seconds']=perf_counter()-s
            s=perf_counter();weights=np.exp(-np.sum((q-z)**2,axis=1)/(2*.7**2));a=(weights[:,None]*v).sum(0)/den;timings['aggregation_seconds']=perf_counter()-s
        elif method=='torch_cpu_exact':
            s=perf_counter();idx=torch.from_numpy(positions);qt=torch.from_numpy(q);z=kcpu.index_select(0,idx);v=vcpu.index_select(0,idx);timings['gather_conversion_seconds']=perf_counter()-s
            s=perf_counter();a=(torch.exp(-((qt-z)**2).sum(-1)/(2*.7**2))[:,None]*v).sum(0)/den;timings['aggregation_seconds']=perf_counter()-s
            s=perf_counter();a=a.numpy();timings['gather_conversion_seconds']+=perf_counter()-s
        elif method=='torch_cuda_exact':
            torch.cuda.synchronize();s=perf_counter();idx=torch.from_numpy(positions).to('cuda:0');qt=torch.from_numpy(q).to('cuda:0');torch.cuda.synchronize();timings['h2d_seconds']=perf_counter()-s
            s=perf_counter();z=kgpu.index_select(0,idx);v=vgpu.index_select(0,idx);torch.cuda.synchronize();timings['gather_conversion_seconds']=perf_counter()-s
            s=perf_counter();at=(torch.exp(-((qt-z)**2).sum(-1)/(2*.7**2))[:,None]*v).sum(0)/den;torch.cuda.synchronize();timings['aggregation_seconds']=perf_counter()-s
            s=perf_counter();a=at.cpu().numpy();torch.cuda.synchronize();timings['d2h_seconds']=perf_counter()-s
        else:
            aggregation_budget=0. if epsilon==0 else float(np.nextafter(epsilon/session._amplification*den,0.))
            s=perf_counter();r=active_store.query_certified(q,datetime_utc(t),aggregation_budget);a=r.aggregate/den;raw_bound=r.aggregate_bound;stats=asdict(r.stats);timings['aggregation_seconds']=perf_counter()-s
        s=perf_counter();cor=model.text_correction(tx,torch.tensor(a[None],dtype=torch.float64));prediction=(model.numeric_forecast(tx)+cor).numpy();timings['forecast_final_seconds']=perf_counter()-s
        timings['end_to_end_seconds']=perf_counter()-total
        bound=session._amplification*raw_bound/den
        refa,refp,refn=references[t];assert n==refn
        deviation=float(np.linalg.norm(prediction-refp));agg_deviation=float(np.linalg.norm(a-refa))
        tolerance=1e-10+1e-10*float(np.linalg.norm(refp))
        if deviation>bound+tolerance or (epsilon is not None and bound>epsilon):raise AssertionError('same-operator certificate failed')
        return dict(method=method,device='cuda_aggregation_cpu_query_forecast' if method=='torch_cuda_exact' else 'cpu',
            epsilon=epsilon,phase=phase,repeat=repeat,cutoff=t,year=datetime_utc(t).year,legal_documents=n,
            metadata_candidate_rows=len(active_available),query_metadata_cache_bytes=active_available.nbytes+active_starts.nbytes+active_ends.nbytes,
            output_budget=epsilon,raw_aggregation_budget=aggregation_budget,normalized_aggregate_bound=raw_bound/den,
            certified_output_bound=bound,actual_output_deviation=deviation,actual_aggregate_deviation=agg_deviation,
            documents_scanned=stats['documents_scanned'] if stats else n,
            scan_fraction=(stats['documents_scanned'] if stats else n)/max(n,1),
            query_stats=stats,timings=timings)
    records=[];warmup_seconds=0.
    for j,item in enumerate(queries):
        if prefix_mode:prepare_prefix(item[2])
        if j==0:
            for method,e in methods:records.append(execute(*item,method,e,'cold_invocation',0))
        warm_start=perf_counter()
        for _ in range(cfg['system']['warmups']):
            for method,e in methods:execute(*item,method,e,'unrecorded_warmup',0)
        warmup_seconds+=perf_counter()-warm_start
        for repeat in range(cfg['system']['repeats']):
            shift=(repeat+j)%len(methods)
            for method,e in methods[shift:]+methods[:shift]:records.append(execute(*item,method,e,'warm',repeat))
    write(out/'query_records.json',records)
    flat=[{k:v for k,v in r.items() if k not in ('query_stats','timings')}|r['timings']|
          {'stat_'+k:v for k,v in (r['query_stats'] or {}).items()} for r in records]
    fields=list(dict.fromkeys(k for r in flat for k in r))
    with (out/'query_records.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    summary=[]
    for method,e in methods:
        selected=[r for r in records if r['method']==method and r['epsilon']==e and r['phase']=='warm']
        summary.append(dict(method=method,epsilon=e,device=selected[0]['device'],
            timings={k:describe([r['timings'][k] for r in selected]) for k in selected[0]['timings']},
            scan=describe([r['scan_fraction'] for r in selected]),
            nodes=describe([r['query_stats']['nodes_visited'] for r in selected]) if method=='citra_avl' else None))
    memory.update(benchmark_rss=rss(),benchmark_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        index_resident_estimate=index_bytes,metadata_cache_bytes=cached.bytes,
        encoded_numpy_bytes=session.keys_np.nbytes+session.impacts_np.nbytes,
        cuda_allocated=torch.cuda.memory_allocated() if kgpu is not None else 0,
        cuda_peak_allocated=torch.cuda.max_memory_allocated() if kgpu is not None else 0,
        cuda_peak_reserved=torch.cuda.max_memory_reserved() if kgpu is not None else 0)
    maintenance_report=maintenance(session,cfg['system']['maintenance_updates']);write(out/'maintenance.json',maintenance_report)
    memory['peak_including_scratch_maintenance']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write(out/'summary.json',dict(checkpoint=str(checkpoint),model_version=session.model_version,queries=len(queries),archive_mode=args.archive_mode,prefix_builds=prefix_builds,
        query_cutoffs=[t for _,_,t in queries],summary=summary,memory=memory,offline=session.offline_timings,
        extra_representation_cache_seconds=cache_seconds,warmup_seconds=warmup_seconds,
        elapsed_seconds=perf_counter()-started,
        cache_assumption='same frozen canonical CPU float64 key/impact bytes for all methods; metadata arrays shared; CUDA copies cached offline',
        cuda_scope='only aggregation offloaded; history/query encoding and final numeric forecast/correction on CPU for every method',
        cold_scope='first invocation after model/representation/device initialization; not a fresh OS process or cold disk cache',
        prefix_scope='separate as-of store for every cutoff, only publicly accessible records inserted; each cached historical state construction is charged offline, not a new index design' if prefix_mode else 'full 2020-2023 historical replay store, future records excluded from answers but can affect tree geometry',
        memory_scope='RSS/peak covers shared data/model/CPU+GPU caches and the index; index estimate separate; maintenance scratch peak separate',
        attribution='tree time includes traversal, metadata checks, summaries and fsum; counters do not establish exclusive time attribution'))
    print(json.dumps([dict(method=x['method'],epsilon=x['epsilon'],p50_ms=x['timings']['end_to_end_seconds']['p50']*1000) for x in summary],indent=2))


if __name__=='__main__':main()
