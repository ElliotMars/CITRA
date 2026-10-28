"""Comparable training modes; Full Text and certified execution share parameters."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from data_provider.citra_weather import WeatherData, datetime_utc
from citra.learning.inference import FrozenForecastSession
from citra.learning.training import fit, seed_everything, load_checkpoint
from models.CITRA import Model


def evaluate(model, data, config, mode):
    model.eval().requires_grad_(False)
    session = FrozenForecastSession(model, data.corpus)
    windows = data.windows("test", model.config["seq_len"], model.config["horizon"], model.config["target_ids"], config["stride"])
    count = min(len(windows), config["evaluation_limit"]) if config["evaluation_limit"] else len(windows)
    rows, predictions, truths = [], [], []
    certified_errors = {str(eps): [] for eps in config["epsilons"]}
    certified_predictions = {str(eps): [] for eps in config["epsilons"]}
    for i in range(count):
        history, y, cutoff = windows[i]
        full = session.predict(history[None], datetime_utc(cutoff), execution=mode, topk=config["topk"])
        rows.append(dict(cutoff=datetime_utc(cutoff).isoformat(), mode=mode, timings=full.timings))
        predictions.append(full.prediction[0])
        truths.append(y)
        if mode == "full_asof_text":
            for epsilon in config["epsilons"]:
                fast = session.predict(history[None], datetime_utc(cutoff), epsilon, execution="citra_certified")
                error = float(np.linalg.norm(fast.prediction - full.prediction))
                tolerance = 1e-10 + 1e-10 * float(np.linalg.norm(full.prediction))
                if error > fast.certified_bound + tolerance or fast.certified_bound > epsilon:
                    raise AssertionError("prediction certificate violation")
                fast.assert_current()
                row = dict(cutoff=datetime_utc(cutoff).isoformat(), mode="citra_certified", epsilon=epsilon,
                           actual_output_deviation=error, certified_bound=fast.certified_bound,
                           comparison_atol=1e-10, comparison_rtol=1e-10, certificate_scope=fast.certificate_scope,
                           bound_tightness=error / fast.certified_bound if fast.certified_bound else None,
                           timings=fast.timings, query_stats=asdict(fast.query_stats))
                certified_errors[str(epsilon)].append(row)
                certified_predictions[str(epsilon)].append(fast.prediction[0])
    def metrics(pred):
        # Report real observation units separately from normalized certificates.
        targets = model.config["target_ids"]
        errors = (np.stack(pred) - np.stack(truths)) * data.scale[targets]
        return dict(mse=float(np.mean(errors**2)), mae=float(np.mean(np.abs(errors))),
                    per_target={data.manifest["channels"][target]: dict(
                        mse=float(np.mean(errors[:, :, i]**2)), mae=float(np.mean(np.abs(errors[:, :, i]))))
                                for i, target in enumerate(targets)})
    report = dict(evaluated_windows=count, available_test_windows=len(windows), queries=rows,
                  offline_timings=session.offline_timings, model_version=session.model_version,
                  encoder_version=session.encoder_version, impact_version=session.impact_version,
                  index_generation=session.index_generation, certified=certified_errors,
                  certificate_units="normalized model outputs, one sample all horizons/targets L2")
    if data.manifest["is_fixture"]:
        report["real_forecast_metrics"] = "omitted: synthetic integration smoke only"
    else:
        report["metrics"] = metrics(predictions)
        report["certified_metrics"] = {eps: metrics(pred) for eps, pred in certified_predictions.items() if pred}
    for name, records in [(mode, rows)] + [(f"citra_epsilon_{eps}", rows) for eps, rows in certified_errors.items()]:
        if records:
            times = [row["timings"]["total_inference_seconds"] for row in records]
            report.setdefault("latency", {})[name] = dict(p50_seconds=float(np.percentile(times, 50)),
                                                         p95_seconds=float(np.percentile(times, 95)),
                                                         total_seconds=sum(times), queries=len(times))
    return report


def run_experiment(config, output, device="cpu", mode="all", checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; GPU experiment was not run")
    data = WeatherData(config["data_path"], allow_simulated=config["allow_simulated"])
    resolved = dict(config, model=dict(config["model"], base_dim=data.manifest["base_dim"],
                                     metadata_dim=data.manifest["metadata_dim"], channels=data.values.shape[1]))
    channels = resolved["model"]["channels"]
    if resolved["model"].get("target_ids") is None:
        resolved["model"]["target_ids"] = list(range(channels))
    (output / "resolved_config.json").write_text(json.dumps(dict(config=resolved, manifest=data.manifest,
        corpus_fingerprint=data.corpus.fingerprint, data_fingerprint=data.fingerprint,
        scaler=dict(mean=data.mean.tolist(), scale=data.scale.tolist()),
        device=device, torch=torch.__version__, numpy=np.__version__), indent=2) + "\n")
    modes = ["ts_only", "semantic_topk", "full_asof_text"] if mode == "all" else [mode]
    results = dict(is_fixture=data.manifest["is_fixture"], device=device, models={},
                   availability_policy="simulated scenario" if config["allow_simulated"] else "verified only")
    for training_mode in modes:
        seed_everything(config["seed"])
        if checkpoint:
            model, state = load_checkpoint(checkpoint, data, device)
            required_mode = "full_asof_text" if training_mode == "citra_certified" else training_mode
            if state["training_mode"] != required_mode:
                raise ValueError("checkpoint training mode differs from requested evaluation")
            training = dict(checkpoint=str(checkpoint), loaded=True)
        else:
            if training_mode == "citra_certified":
                raise ValueError("citra_certified requires a Full As-of checkpoint")
            model = Model(**resolved["model"]).double().to(device)
            model, training = fit(model, data, config, training_mode, output)
        inference_mode = "full_asof_text" if training_mode == "citra_certified" else training_mode
        results["models"][training_mode] = dict(training=training,
            evaluation=evaluate(model, data, config, inference_mode))
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results
