"""Train document classifiers with grouped holdout validation and best checkpoints."""

import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.evaluate import export_evaluation
from scripts.model import DocumentDataset, build_model
from scripts.preprocess import CLASSES, ROOT, data_directory, make_split, metadata_digest, read_metadata, verified_cache


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch and disable nondeterministic cuDNN tuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed Python and NumPy from each DataLoader worker's PyTorch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested="auto"):
    """Choose CUDA when available and reject unavailable explicitly requested CUDA."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: install a matching CUDA-enabled PyTorch build")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def make_loader(dataset, config, shuffle, device):
    """Construct a seeded loader with optional pinned memory and Windows workers."""
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=config["workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config["seed"]),
                      persistent_workers=config["workers"] > 0)


def evaluate_model(model, loader, device):
    """Collect probabilities and unsmoothed cross entropy without gradient updates."""
    model.eval()
    probability_batches, targets = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            probability_batches.append(logits.softmax(dim=1).cpu().numpy())
            targets.extend(labels.cpu().tolist())
    probabilities = np.concatenate(probability_batches)
    predicted = probabilities.argmax(axis=1)
    return probabilities, {"val_loss": total_loss / len(targets),
                           "accuracy": float(accuracy_score(targets, predicted)),
                           "macro_f1": float(f1_score(targets, predicted, labels=range(16),
                                                       average="macro", zero_division=0))}


def validate_initial_checkpoint(payload, model_name, digest, validation_ids):
    """Reject warm starts whose dataset, label mapping, architecture, or holdout differs."""
    if payload.get("classes") != CLASSES or payload.get("metadata_digest") != digest:
        raise ValueError("Initial checkpoint has a different dataset or class mapping")
    if payload["config"]["model"] != model_name:
        raise ValueError("Initial checkpoint architecture differs from the requested model")
    previous_ids = payload.get("validation_ids", [])
    if len(previous_ids) != len(validation_ids) or set(previous_ids) != set(validation_ids):
        raise ValueError("Initial checkpoint holdout differs; warm-start validation would risk leakage")


def run_training(config_path=None, data_dir=None, device="auto", epochs=None, run_name=None):
    """Train on the training split, select on validation, and export a complete run."""
    config_path = Path(config_path or ROOT / "configs/baseline.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if epochs is not None:
        config["epochs"] = epochs
    if run_name is not None:
        config["run_name"] = run_name
    if Path(config["run_name"]).name != config["run_name"] or config["run_name"] in {".", ".."}:
        raise ValueError("run_name must be a simple directory name")
    if config["epochs"] < 1 or config["selection_metric"] not in {"accuracy", "macro_f1"}:
        raise ValueError("epochs must be positive and selection_metric accuracy or macro_f1")
    data_dir = data_directory(data_dir)
    train, _, _ = read_metadata(data_dir)
    cache_dir = verified_cache(data_dir, config["image_size"])
    audit = pd.read_csv(cache_dir / "train_audit.csv")
    train_index, val_index = make_split(train, audit, config["seed"], config["validation_fraction"])
    output = ROOT / "reports" / config["run_name"]
    checkpoint_dir = ROOT / "checkpoint" / config["run_name"]
    if (checkpoint_dir / "best.pt").exists():
        raise FileExistsError("Run already has a checkpoint. Choose a new --run-name to preserve it.")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    torch.set_num_threads(config["threads"])
    selected_device = select_device(device)
    torch.hub.set_dir(str(ROOT / "checkpoint/pretrained"))
    model = build_model(config["model"], config["pretrained"]).to(selected_device)
    initialization = None
    if config.get("init_checkpoint"):
        initial_path = ROOT / config["init_checkpoint"]
        initial = torch.load(initial_path, map_location=selected_device, weights_only=True)
        validate_initial_checkpoint(initial, config["model"], metadata_digest(data_dir),
                                    train.iloc[val_index].id.tolist())
        model.load_state_dict(initial["state_dict"])
        initialization = {"path": config["init_checkpoint"],
                          "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                          "source_epoch": initial["epoch"], "optimizer_reset": True}
        del initial
    labels = train.label.to_numpy()
    train_set = DocumentDataset(cache_dir / "train.npy", train_index, labels, config["model"],
                                augment=True, geometry=config.get("geometry"))
    val_set = DocumentDataset(cache_dir / "train.npy", val_index, labels, config["model"])
    train_loader = make_loader(train_set, config, True, selected_device)
    val_loader = make_loader(val_set, config, False, selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    amp_enabled = config["amp"] and selected_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    split = train[["id", "label"]].copy()
    split["split"] = "excluded_unreadable"
    split.loc[train_index, "split"] = "train"
    split.loc[val_index, "split"] = "validation"
    split.to_csv(output / "split.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pandas", "scikit-learn", "Pillow", "matplotlib")}
    if config["model"] in {"efficientnet_b0", "resnet50", "convnext_tiny"}:
        versions["torchvision"] = importlib.metadata.version("torchvision")
    run = {"config": config, "device": str(selected_device), "versions": versions,
           "initialization": initialization,
           "device_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU",
           "train_rows": len(train_index), "validation_rows": len(val_index),
           "metadata_digest": metadata_digest(data_dir),
           "parameter_count": sum(p.numel() for p in model.parameters())}
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    history, best, stale = [], -1.0, 0
    print(json.dumps(run, indent=2), flush=True)
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for batch_number, (images, targets) in enumerate(train_loader, start=1):
            images, targets = images.to(selected_device), targets.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=selected_device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(targets)
            seen += len(targets)
            if batch_number % 200 == 0:
                print(f"Epoch {epoch}: {seen}/{len(train_set)} images; "
                      f"loss={loss_sum / seen:.4f}; elapsed={time.perf_counter() - started:.0f}s", flush=True)
        _, metrics = evaluate_model(model, val_loader, selected_device)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, **metrics,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        scheduler.step()
        score = metrics[config["selection_metric"]]
        if score > best:
            best, stale = score, 0
            payload = {"state_dict": model.state_dict(), "config": config, "classes": CLASSES,
                       "initialization": initialization,
                       "epoch": epoch, "metrics": metrics, "metadata_digest": run["metadata_digest"],
                       "validation_ids": train.iloc[val_index].id.tolist()}
            torch.save(payload, checkpoint_dir / "best.partial.pt")
            (checkpoint_dir / "best.partial.pt").replace(checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    saved = torch.load(checkpoint_dir / "best.pt", map_location=selected_device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    probabilities, _ = evaluate_model(model, val_loader, selected_device)
    np.savez_compressed(checkpoint_dir / "validation_probabilities.npz",
                        ids=train.iloc[val_index].id.to_numpy(dtype=str), probabilities=probabilities,
                        checkpoint_sha256=hashlib.sha256((checkpoint_dir / "best.pt").read_bytes()).hexdigest())
    result = export_evaluation(train.iloc[val_index], probabilities, output, history)
    result["best_epoch"] = saved["epoch"]
    result["selection_metric"] = config["selection_metric"]
    majority = int(train.iloc[train_index].label.mode().iloc[0])
    result["majority_baseline_accuracy"] = float(train.iloc[val_index].label.eq(majority).mean())
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)
    return result


def main():
    """Expose configurable training without changing notebook source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    run_training(args.config, args.data_dir, args.device, args.epochs, args.run_name)


if __name__ == "__main__":
    main()
