"""Differentiable full aggregation training and deterministic chronological validation."""
import json
from pathlib import Path
import random
from time import perf_counter
import numpy as np
import torch
from models.CITRA import Model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_tensors(dataset, indices, device, include_text=True):
    batch = [dataset[int(i)] for i in indices]
    x = torch.tensor(np.stack([r[0] for r in batch]), dtype=torch.float64, device=device)
    y = torch.tensor(np.stack([r[1] for r in batch]), dtype=torch.float64, device=device)
    cutoffs = [r[2] for r in batch]
    mask = (torch.tensor(dataset.data.corpus.visibility(cutoffs), dtype=torch.bool, device=device)
            if include_text else None)
    return x, y, mask


def fit(model, data, config, mode, output):
    if mode not in ("ts_only", "semantic_topk", "full_asof_text", "shuffled_text"):
        raise ValueError("CITRA inference shares the Full As-of checkpoint; it is not trained separately")
    device = next(model.parameters()).device
    targets = model.config["target_ids"]
    train = data.windows("train", model.config["seq_len"], model.config["horizon"], targets, config["stride"])
    validation = data.windows("validation", model.config["seq_len"], model.config["horizon"], targets, config["stride"])
    if mode != "ts_only" and not data.corpus.visibility([train[i][2] for i in range(len(train))]).any():
        raise ValueError("no text is historically accessible in training; verify source times instead of inventing availability")
    if mode == "shuffled_text" and not getattr(data, "control_kind", None):
        raise ValueError("shuffled_text requires explicitly constructed causal control data")
    base, metadata = data.corpus.tensors(device, torch.float64) if mode != "ts_only" else (None, None)
    forward_mode = "full_asof_text" if mode == "shuffled_text" else mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = np.random.default_rng(config["seed"])
    logs, best = [], float("inf")
    checkpoint = Path(output) / f"{mode}.pt"
    for epoch in range(config["epochs"]):
        model.train()
        start, losses = perf_counter(), []
        order = generator.permutation(len(train))
        for begin in range(0, len(order), config["batch_size"]):
            idx = order[begin:begin + config["batch_size"]]
            x, y, mask = batch_tensors(train, idx, device, include_text=mode != "ts_only")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, base, metadata, mask, mode=forward_mode, topk=config["topk"])
            loss = (prediction - y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            model.fusion.constrain(config["fusion_norm_limit"])
            losses.append((float(loss.detach()), len(idx)))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for begin in range(0, len(validation), config["batch_size"]):
                idx = range(begin, min(len(validation), begin + config["batch_size"]))
                x, y, mask = batch_tensors(validation, idx, device, include_text=mode != "ts_only")
                prediction = model(x, base, metadata, mask, mode=forward_mode, topk=config["topk"])
                validation_losses.append((float((prediction - y).square().mean()), len(idx)))
        training_loss = sum(loss * count for loss, count in losses) / len(train)
        validation_loss = sum(loss * count for loss, count in validation_losses) / len(validation)
        if not np.isfinite(validation_loss):
            raise FloatingPointError("nonfinite validation loss")
        logs.append(dict(epoch=epoch + 1, training_loss=training_loss, validation_loss=validation_loss,
                         elapsed_seconds=perf_counter() - start, is_fixture=data.manifest["is_fixture"]))
        if validation_loss < best:
            best = validation_loss
            torch.save(dict(state_dict=model.state_dict(), model_config=model.config, training_mode=mode,
                            config=config, data_fingerprint=data.numeric_fingerprint if mode == "ts_only" else data.fingerprint,
                            fingerprint_scope="numeric_only" if mode == "ts_only" else "numeric_and_text",
                            scaler=dict(mean=data.mean.tolist(), scale=data.scale.tolist()),
                            train_end=data.manifest["train_end"], validation_end=data.manifest["validation_end"],
                            best_epoch=epoch + 1), checkpoint)
    (Path(output) / f"{mode}_training.json").write_text(json.dumps(logs, indent=2, allow_nan=False) + "\n")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model, dict(checkpoint=str(checkpoint), epochs=logs, best_epoch=state["best_epoch"],
                       training_windows=len(train), validation_windows=len(validation),
                       training_seconds=sum(row["elapsed_seconds"] for row in logs))


def load_checkpoint(path, data, device):
    state = torch.load(path, map_location=device, weights_only=True)
    expected_fingerprint = data.numeric_fingerprint if state.get("fingerprint_scope") == "numeric_only" else data.fingerprint
    if (state["data_fingerprint"] != expected_fingerprint
            or state["scaler"] != dict(mean=data.mean.tolist(), scale=data.scale.tolist())
            or state["train_end"] != data.manifest["train_end"]
            or state["validation_end"] != data.manifest["validation_end"]):
        raise ValueError("checkpoint corpus/scaler/time splits differ; explicit retraining is required")
    model = Model(**state["model_config"]).double().to(device)
    model.load_state_dict(state["state_dict"])
    return model, state
