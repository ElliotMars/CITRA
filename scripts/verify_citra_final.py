"""Independent artifact consistency audit, not historical-availability verification."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from citra.final_diagnosis import protocol_config_path,paired_block_bootstrap
import yaml


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg=yaml.safe_load(protocol_config_path().read_text());root=Path(cfg['output']);reg=read(root/'registration.json')
    assert digest(protocol_config_path())==reg['config_sha256']
    lock=read(root/'semantics/methods_frozen.json')
    checkpoints=reg['checkpoints']|lock['new_checkpoints']
    for path,expected in checkpoints.items():assert digest(path)==expected,path
    manifest=read(root/'new_year_manifest.json')
    for path,expected in manifest['source_files'].items():assert digest(path)==expected,path
    evaluations=read(root/'semantics/evaluations.json');summary=read(root/'semantics/summary.json')
    dates_by_year={};losses={};arrays_checked=0
    for r in evaluations['rows']:
        with np.load(root/'semantics'/r['kind']/f"seed_{r['seed']}"/f"year_{r['year']}.npz") as a:
            errors=a['prediction']-a['truth'];mse=float(np.mean(errors**2));mae=float(np.mean(np.abs(errors)))
            np.testing.assert_allclose([mse,mae],[r['text_on']['normalized_mse'],r['text_on']['normalized_mae']],rtol=1e-12,atol=1e-12)
            for i,target in enumerate(('temperature_F','wind_speed_knots')):
                physical=errors[:,:,i]*r['scaler']['scale'][i]
                np.testing.assert_allclose([np.mean(physical**2),np.mean(np.abs(physical))],
                    [r['text_on']['per_target'][target]['mse'],r['text_on']['per_target'][target]['mae']],rtol=1e-12,atol=1e-12)
            if r['year'] not in dates_by_year:dates_by_year[r['year']]=a['cutoff'].copy()
            np.testing.assert_array_equal(a['cutoff'],dates_by_year[r['year']])
            with np.load(root/'semantics/ts_only'/f"seed_{r['seed']}"/f"year_{r['year']}.npz") as ts:
                np.testing.assert_array_equal(a['numeric'],ts['prediction']);np.testing.assert_array_equal(a['truth'],ts['truth'])
            losses[r['kind'],r['year'],r['seed']]=np.mean(errors**2,axis=(1,2))
            arrays_checked+=1
    for g in summary['groups']:
        rs=[r for r in evaluations['rows'] if r['kind']==g['kind'] and r['year']==g['year']]
        for metric in ('normalized_mse','normalized_mae'):
            np.testing.assert_allclose([np.mean([r['text_on'][metric] for r in rs]),np.std([r['text_on'][metric] for r in rs],ddof=1)],
                [g['metrics'][metric]['mean'],g['metrics'][metric]['std']],rtol=1e-12,atol=1e-12)
    intervals=0
    for r in summary['paired']:
        if r['metric']!='normalized_mse':continue
        control=r['contrast'].removeprefix('real_text minus ')
        delta=np.stack([losses['real_text',r['year'],s]-losses[control,r['year'],s] for s in cfg['seeds']])
        dates=dates_by_year[r['year']].astype('datetime64[s]').astype('datetime64[D]')
        check=paired_block_bootstrap(dates,delta,r['block_days'],r['repeats'],r['seed'])
        for key in ('mean_difference','ci95_low','ci95_high','improvement_date_fraction'):
            np.testing.assert_allclose(check[key],r[key],rtol=1e-12,atol=1e-12)
        intervals+=1
    timing_runs=0
    for mode in ('system','system_asof'):
        records=read(root/mode/'query_records.json');report=read(root/mode/'summary.json')
        for row in records:
            assert row['actual_output_deviation']<=row['certified_output_bound']+1e-9
            if row['epsilon'] is not None:assert row['certified_output_bound']<=row['epsilon']
            if mode=='system_asof':
                prefix=next(p for p in report['prefix_builds'] if p['cutoff']==row['cutoff'])
                assert row['metadata_candidate_rows']==prefix['past_records']
            timing_runs+=1
        for group in report['summary']:
            rs=[r for r in records if r['phase']=='warm' and r['method']==group['method'] and r['epsilon']==group['epsilon']]
            for key in ('p50','p95','p99','max'):
                x=[r['timings']['end_to_end_seconds'] for r in rs]
                actual=max(x) if key=='max' else np.percentile(x,int(key[1:]))
                np.testing.assert_allclose(actual,group['timings']['end_to_end_seconds'][key],rtol=1e-12,atol=1e-12)
    cases=0;synthetic_queries=0
    for n in cfg['scaling']['sizes']:
        for scenario in cfg['scaling']['scenarios']:
            case=read(root/'scaling'/f'{scenario}_{n}.json');assert case['synthetic'] and case['legal_n']==n
            for row in case['records']:
                assert row['actual_output_deviation']<=row['certified_bound']+1e-9
                if row['epsilon'] is not None:assert row['certified_bound']<=row['epsilon']
                synthetic_queries+=1
            cases+=1
    # Preserve the exact training-stage versions if optional CLI-only additions
    # were made later. A snapshot is accepted only if its registered SHA matches.
    snapshots={};folder=root/'source_snapshots';folder.mkdir(exist_ok=True)
    for src,expected in lock['code_sha256'].items():
        text=Path(src).read_text();candidate=text
        if 'def protocol_config_path():' in candidate:
            candidate=candidate.replace('import os\nfrom pathlib import Path\n','').replace(
                '\n\ndef protocol_config_path():\n    """Use a copied protocol with a fresh output directory for reproduction."""\n    return Path(os.environ.get(\'CITRA_FINAL_CONFIG\',\'configs/citra_final.yaml\'))\n','')
        elif src.endswith('run_citra_final_semantics.py'):
            candidate=candidate.replace('paired_block_bootstrap,protocol_config_path','paired_block_bootstrap').replace('protocol_config_path()',"Path('configs/citra_final.yaml')")
        if hashlib.sha256(candidate.encode()).hexdigest()==expected:
            dest=folder/(expected[:12]+'_'+Path(src).name);dest.write_text(candidate);snapshots[src]=dict(registered_sha=expected,preserved=True,path=str(dest))
        else:snapshots[src]=dict(registered_sha=expected,preserved=digest(src)==expected,current_sha=digest(src))
    result=dict(artifact_consistency_passed=True,checkpoints_checked=len(checkpoints),raw_files_checked=len(manifest['source_files']),
        prediction_arrays_checked=arrays_checked,paired_primary_intervals_recomputed=intervals,
        real_performance_runs_checked=timing_runs,synthetic_cases_checked=cases,synthetic_query_runs_checked=synthetic_queries,
        training_sources=snapshots,legacy_core_unchanged=read(root/'repository_audit.json')['unchanged_core_vs_commit'],
        note='Verifies saved metrics, hashes, cohorts and certificates, not historical first-publication evidence or IEEE formal rounding bounds')
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
