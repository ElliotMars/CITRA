"""Real chronological evaluation; accuracy and latency are distinct measurements."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from data_provider.citra_weather import WeatherData, datetime_utc
from citra.learning.inference import FrozenForecastSession, synchronize
from citra.learning.training import fit, seed_everything, load_checkpoint, batch_tensors
from models.CITRA import Model

def error_metrics(predictions, truths, data, targets):
    errors = predictions - truths
    physical = errors * data.scale[list(targets)]
    return dict(mse=float(np.mean(physical**2)), mae=float(np.mean(np.abs(physical))),
        normalized_mse=float(np.mean(errors**2)), normalized_mae=float(np.mean(np.abs(errors))),
        aggregate_physical_units="mixed units; use per_target or normalized metrics for comparisons",
        per_target={data.manifest["channels"][target]: dict(mse=float(np.mean(physical[:,:,i]**2)),
            mae=float(np.mean(np.abs(physical[:,:,i])))) for i,target in enumerate(targets)})

@torch.no_grad()
def evaluate(model, data, config, mode, prediction_output=None):
    model.eval().requires_grad_(False)
    numeric_only = mode == "ts_only"
    actual_mode = "full_asof_text" if mode == "shuffled_text" else mode
    session = FrozenForecastSession(model, None if numeric_only else data.corpus,
                                    text_enabled=not numeric_only, build_index=False)
    c = model.config
    from citra.learning.aggregation import AggregationSpec
    spec=AggregationSpec.from_model(model)
    windows = data.windows("test", c["seq_len"], c["horizon"], c["target_ids"], config["stride"])
    count = min(len(windows), config["evaluation_limit"]) if config["evaluation_limit"] else len(windows)
    device = next(model.parameters()).device
    start = perf_counter()
    predictions, truths, cutoffs = [], [], []
    # Cached representations are legal here only because all parameters are
    # frozen and their exact bytes are bound by the session.
    if not numeric_only:
        keys = torch.tensor(session.keys_np.copy(), dtype=torch.float64, device=device)
        impacts = torch.tensor(session.impacts_np.copy(), dtype=torch.float64, device=device)
    for begin in range(0, count, config["batch_size"]):
        idx = range(begin, min(count, begin + config["batch_size"]))
        x, y, mask = batch_tensors(windows, idx, device, include_text=not numeric_only,aggregation_spec=spec)
        numerical = model.numeric_forecast(x)
        if numeric_only:
            forecast = numerical
        else:
            q = model.encode_query(x)
            aggregate = model.aggregate(q, keys, impacts, mask, actual_mode, config["topk"])
            forecast = numerical + model.text_correction(x,aggregate)
        predictions.append(forecast.cpu().numpy())
        truths.append(y.cpu().numpy())
        cutoffs.extend(windows[i][2] for i in idx)
    synchronize(model)
    accuracy_seconds = perf_counter() - start
    predictions, truths = np.concatenate(predictions), np.concatenate(truths)
    if prediction_output:
        np.savez_compressed(prediction_output, prediction_normalized=predictions, truth_normalized=truths,
                            cutoff=np.asarray(cutoffs,dtype=np.int64))
    report = dict(evaluated_windows=count, available_test_windows=len(windows), accuracy_evaluation_seconds=accuracy_seconds,
        model_version=session.model_version, encoder_version=session.encoder_version, impact_version=session.impact_version,
        index_generation=session.index_generation, offline_timings=session.offline_timings,
        availability_protocol=data.availability_protocol, text_delay_hours=data.manifest.get("text_delay_hours"),
        numeric_availability_protocol=data.manifest.get("numeric_availability_protocol"),
        numeric_delay_hours=data.manifest.get("numeric_delay_hours"), is_fixture=data.manifest["is_fixture"],
        queries=[], certified={}, certificate_units="normalized single-sample [H,C_target] L2",
        control_kind=getattr(data,"control_kind",None),
        aggregation_semantics=dict(kind=spec.kind,window_days=spec.window_days,gated=c.get("gated",False)),
        accuracy_note="all reported test windows; batched accuracy timing is not per-query latency")
    if data.manifest["is_fixture"]:
        report["real_forecast_metrics"]="omitted: synthetic integration smoke only"
    else:
        report["metrics"]=error_metrics(predictions,truths,data,c["target_ids"])
    n_profile = min(count,config.get("profile_queries",12))
    if n_profile == 0:
        return report
    selected = np.unique(np.linspace(0,count-1,n_profile,dtype=int))
    references={}
    repeats=config.get("profile_repeats",1)
    for i in selected:
        x,_,t=windows[int(i)]
        cutoff=datetime_utc(t)
        timings=[]
        for repeat in range(repeats):
            result=session.predict(x[None],cutoff,execution=actual_mode,topk=config["topk"])
            timings.append(result.timings)
        np.testing.assert_allclose(result.prediction[0],predictions[int(i)],atol=1e-10,rtol=1e-10)
        references[int(i)]=result.prediction
        report["queries"].append(dict(window_index=int(i),cutoff=cutoff.isoformat(),mode=mode,
            timings=timings,first_timings=timings[0],execution="vectorized_exact" if actual_mode=="full_asof_text" else actual_mode))
    if mode=="full_asof_text" and config.get("certified_profile",True):
        session.prepare_index() # charged offline, never silently built in an online call
        nodes=session.store.index.nodes
        radii=np.asarray([n.radius for n in nodes])
        report["index_geometry"]=dict(nodes=len(nodes),version_intervals=session.store.index.root.count if nodes else 0,
            radius_min=float(radii.min()) if len(radii) else 0.,radius_p50=float(np.percentile(radii,50)) if len(radii) else 0.,
            radius_p95=float(np.percentile(radii,95)) if len(radii) else 0.,radius_max=float(radii.max()) if len(radii) else 0.,
            resident_bytes_estimate=session.store.index.resident_bytes)
        from citra.learning.diagnostics import local_bounds, maintenance
        report["local_bound_diagnostics"]=[]
        for i in selected:
            x,_,t=windows[int(i)]
            report["local_bound_diagnostics"].append(dict(window_index=int(i),
                **local_bounds(session,x[None],datetime_utc(t))))
        for epsilon in config["epsilons"]:
            rows=[]
            for i in selected:
                x,_,t=windows[int(i)]
                cutoff=datetime_utc(t)
                timings=[]
                for repeat in range(repeats):
                    fast=session.predict(x[None],cutoff,epsilon,execution="citra_certified")
                    timings.append(fast.timings)
                error=float(np.linalg.norm(fast.prediction-references[int(i)]))
                tolerance=1e-10+1e-10*float(np.linalg.norm(references[int(i)]))
                if error>fast.certified_bound+tolerance or fast.certified_bound>epsilon:
                    raise AssertionError("prediction certificate violation")
                visible=int(spec.mask(data.corpus,[t]).sum())
                stats=asdict(fast.query_stats)
                visited=max(stats["nodes_visited"],1)
                rows.append(dict(window_index=int(i),cutoff=cutoff.isoformat(),epsilon=epsilon,
                    actual_output_deviation=error,certified_bound=fast.certified_bound,aggregate_bound=fast.aggregate_bound,
                    bound_tightness=error/fast.certified_bound if fast.certified_bound else None,
                    comparison_atol=1e-10,comparison_rtol=1e-10,certificate_scope=fast.certificate_scope,
                    visible_documents=visible,scan_fraction=stats["documents_scanned"]/visible if visible else 0.,
                    partial_node_fraction=stats["temporal_partial_nodes"]/visited,
                    visible_node_fraction=stats["temporal_visible_nodes"]/visited,
                    invisible_node_fraction=stats["temporal_nodes_pruned"]/visited,
                    timings=timings,first_timings=timings[0],query_stats=stats))
            report["certified"][str(epsilon)]=rows
        report["index_maintenance"]=maintenance(session)
        session.offline_timings["index_maintenance_seconds"]=report["index_maintenance"]["total_seconds"]
    report["offline_timings"]=session.offline_timings
    report["latency"]={}
    groups=[(mode,report["queries"])]+[(f"citra_epsilon_{e}",r) for e,r in report["certified"].items()]
    for name,rows in groups:
        timings=[t for row in rows for t in row["timings"]]
        values=[t["total_inference_seconds"] for t in timings]
        report["latency"][name]=dict(p50_seconds=float(np.percentile(values,50)),
            p95_seconds=float(np.percentile(values,95)),total_seconds=sum(values),queries=len(values),
            components_mean_seconds={k:float(np.mean([t[k] for t in timings])) for k in timings[0]})
    return report

def run_experiment(config,output,device="cpu",mode="all",checkpoint=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config["threads"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    load_start=perf_counter()
    data=WeatherData(config["data_path"],allow_simulated=config.get("allow_simulated",False),
        availability_protocol=config.get("availability_protocol"),text_delay_hours=config.get("text_delay_hours"),
        load_text=mode!="ts_only")
    load_seconds=perf_counter()-load_start
    resolved=dict(config,model=dict(config["model"],base_dim=data.manifest["base_dim"],
        metadata_dim=data.manifest["metadata_dim"],channels=data.values.shape[1]))
    if resolved["model"].get("target_ids") is None:
        resolved["model"]["target_ids"]=list(range(data.values.shape[1]))
    (output/"resolved_config.json").write_text(json.dumps(dict(config=resolved,manifest=data.manifest,
        corpus_fingerprint=data.corpus.fingerprint,data_fingerprint=data.fingerprint,numeric_fingerprint=data.numeric_fingerprint,
        scaler=dict(mean=data.mean.tolist(),scale=data.scale.tolist()),device=device,torch=str(torch.__version__),
        numpy=np.__version__),indent=2)+"\n")
    modes=["ts_only","semantic_topk","full_asof_text","shuffled_text"] if mode=="all" else [mode]
    results=dict(is_fixture=data.manifest["is_fixture"],device=device,models={},data_load_seconds=load_seconds,
        availability_policy=data.availability_protocol,text_delay_hours=data.manifest.get("text_delay_hours"),
        numeric_availability_protocol=data.manifest.get("numeric_availability_protocol"),
        numeric_delay_hours=data.manifest.get("numeric_delay_hours"))
    for training_mode in modes:
        seed_everything(config["seed"])
        working=data.with_causal_shuffle(config["seed"]) if training_mode=="shuffled_text" else data
        if checkpoint:
            model,state=load_checkpoint(checkpoint,working,device)
            required="full_asof_text" if training_mode=="citra_certified" else training_mode
            if state["training_mode"]!=required:
                raise ValueError("checkpoint training mode mismatch")
            training=dict(checkpoint=str(checkpoint),loaded=True)
        else:
            if training_mode=="citra_certified":
                raise ValueError("certified evaluation needs a shared Full As-of checkpoint")
            model=Model(**resolved["model"]).double().to(device)
            model,training=fit(model,working,config,training_mode,output)
        inference_mode="full_asof_text" if training_mode=="citra_certified" else training_mode
        evaluation=evaluate(model,working,config,inference_mode,output/f"{training_mode}_predictions.npz")
        results["models"][training_mode]=dict(training=training,evaluation=evaluation)
        (output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    return results
