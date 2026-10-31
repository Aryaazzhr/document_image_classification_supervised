"""Fit a document layout classifier on frozen full-page and header feature maps."""

import hashlib
import json
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from scripts.evaluate import export_evaluation
from scripts.model import build_model
from scripts.preprocess import ROOT, CLASSES, data_directory, read_metadata, metadata_digest, resolve_image
from scripts.train import seed_everything, validate_initial_checkpoint


class PageViews(Dataset):
    """Decode two deterministic views of a TIFF identified by its metadata row."""

    def __init__(self, frame, data_dir, config):
        """Keep ordered file names and the crop geometry for each view."""
        self.files = frame.filename.tolist()
        self.data_dir = data_dir
        self.config = config

    def __len__(self):
        """Return the number of pages in the partition."""
        return len(self.files)

    def __getitem__(self, index):
        """Keep the full page and crop its upper portion before downsampling."""
        with Image.open(resolve_image(self.data_dir, self.files[index])) as original:
            page = ImageOps.exif_transpose(original).convert("L")
        header = page.crop((0, 0, page.width, max(1, round(page.height * self.config["header_fraction"]))))
        views = []
        for source, size in ((page, self.config["page_wh"]), (header, self.config["header_wh"])):
            pixels = ImageOps.pad(source, tuple(size), method=Image.Resampling.BILINEAR, color=255)
            views.append(torch.from_numpy(np.array(pixels)))
        return views


def embedding(backbone, pixels, grids, mean, std):
    """Pool intermediate and final spatial maps while retaining their grid positions."""
    values = (pixels[:, None].to(mean.device, dtype=torch.float32) / 255 - mean) / std
    parts = []
    for index, stage in enumerate(backbone.features):
        values = stage(values)
        if index in (5, 7):
            grid = grids[0 if index == 5 else 1]
            cells = F.adaptive_avg_pool2d(values.float(), grid).permute(0, 2, 3, 1)
            parts.append(F.layer_norm(cells, (cells.shape[-1],)).flatten(1))
    return torch.cat(parts, dim=1)


def cached_features(name, frame, backbone, config, folder, identity, device):
    """Cache embeddings atomically and verify their source, view geometry and ID order."""
    key = {"version": 1, "source": identity, "ids": hashlib.sha256("\n".join(frame.id).encode()).hexdigest(),
           "page": config["page_wh"], "header": config["header_wh"], "fraction": config["header_fraction"]}
    file = folder / f"{name}_features.npy"
    manifest = file.with_suffix(".json")
    if file.exists() and manifest.exists() and json.loads(manifest.read_text()) == key:
        cached = np.load(file, mmap_mode="r")
        if cached.shape != (len(frame), 9600) or cached.dtype != np.float16:
            raise ValueError(f"Unexpected cached feature dimensions or dtype for {name}")
        return cached
    loader = DataLoader(PageViews(frame, data_directory(), config), batch_size=config["extract_batch_size"],
                        num_workers=config["workers"], shuffle=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    partial = folder / f"{name}_features.partial.npy"
    matrix, offset = None, 0
    with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for page, header in loader:
            full = embedding(backbone, page, ((3, 3), (2, 2)), mean, std)
            top = embedding(backbone, header, ((2, 3), (1, 1)), mean, std)
            batch = torch.cat((full, top), dim=1).half().cpu().numpy()
            if not np.isfinite(batch).all():
                raise ValueError("Nonfinite spatial embedding")
            if matrix is None:
                matrix = np.lib.format.open_memmap(partial, mode="w+", dtype="float16",
                                                    shape=(len(frame), batch.shape[1]))
            matrix[offset:offset + len(batch)] = batch
            offset += len(batch)
            if offset % (config["extract_batch_size"] * 100) == 0:
                print(f"Extracting {name}: {offset}/{len(frame)}", flush=True)
    assert offset == len(frame)
    matrix.flush()
    del matrix
    manifest.unlink(missing_ok=True)
    partial.replace(file)
    manifest.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return np.load(file, mmap_mode="r")


def predict_head(head, features):
    """Predict in stable metadata order without dropout."""
    head.eval()
    with torch.inference_mode():
        return torch.cat([head(batch.float()).softmax(1) for batch in features.split(256)]).cpu().numpy()


def run():
    """Train the head using training IDs only and export the selected epoch's probabilities."""
    config = json.loads((ROOT / "configs/spatial_layout.json").read_text())
    folder, report = [ROOT / kind / config["run_name"] for kind in ("checkpoint", "reports")]
    folder.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    seed_everything(config["seed"])
    data_dir = data_directory()
    train, test, _ = read_metadata(data_dir)
    split = pd.read_csv(ROOT / "reports" / config["source"] / "split.csv").set_index("id").loc[train.id]
    assert np.array_equal(split.label, train.label)
    split.reset_index().to_csv(report / "split.csv", index=False)
    fit = train.loc[split.split.to_numpy() == "train"].reset_index(drop=True)
    val = train.loc[split.split.to_numpy() == "validation"].reset_index(drop=True)
    digest = metadata_digest(data_dir)
    source = ROOT / "checkpoint" / config["source"] / "best.pt"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    saved = torch.load(source, map_location="cpu", weights_only=True)
    validate_initial_checkpoint(saved, "convnext_tiny", digest, val.id.tolist())
    backbone = build_model("convnext_tiny", pretrained=False)
    backbone.load_state_dict(saved["state_dict"])
    del saved
    backbone.to(device).eval().requires_grad_(False)
    matrices = [cached_features(name, frame, backbone, config, folder, source_hash + digest, device)
                for name, frame in (("train", fit), ("validation", val), ("test", test))]
    del backbone
    train_x, val_x, test_x = [torch.from_numpy(np.array(matrix)).to(device) for matrix in matrices]
    labels = torch.tensor(fit.label.to_numpy(), device=device)
    width = train_x.shape[1]
    head = nn.Sequential(nn.Dropout(0.25), nn.Linear(width, config["hidden"]), nn.LayerNorm(config["hidden"]),
                         nn.GELU(), nn.Dropout(0.35), nn.Linear(config["hidden"], 16)).to(device)
    checkpoint = folder / "best.pt"
    run_info = {"config": config, "metadata_digest": digest, "source_checkpoint_sha256": source_hash,
                "feature_width": width, "device": str(device), "train_rows": len(fit), "validation_rows": len(val)}
    fit_required = True
    if checkpoint.exists():
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_initial_checkpoint(stored, config["model"], digest, val.id.tolist())
        if stored["config"] != config or stored["source_checkpoint_sha256"] != source_hash:
            raise ValueError("Existing checkpoint belongs to a different spatial experiment")
        history = pd.read_csv(report / "history.csv").to_dict("records")
        fit_required = (len(history) != config["epochs"] or
                        stored["epoch"] != max(history, key=lambda row: row["accuracy"])["epoch"])
    if fit_required:
        optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
        history, best = [], -1
        for epoch in range(1, config["epochs"] + 1):
            head.train()
            started, loss_sum = time.perf_counter(), 0.0
            for indices in torch.randperm(len(fit), device=device).split(config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(train_x[indices].float()), labels[indices], label_smoothing=config["label_smoothing"])
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1)
                optimizer.step()
                loss_sum += loss.item() * len(indices)
            probs = predict_head(head, val_x)
            pred = probs.argmax(1)
            acc = float((pred == val.label).mean())
            row = {"epoch": epoch, "train_loss": loss_sum / len(fit), "accuracy": acc,
                   "macro_f1": f1_score(val.label, pred, average="macro"),
                   "val_loss": float(-np.log(probs[np.arange(len(val)), val.label].clip(1e-10)).mean()),
                   "seconds": time.perf_counter() - started}
            scheduler.step()
            history.append(row)
            pd.DataFrame(history).to_csv(report / "history.csv", index=False)
            print(f"Epoch {epoch}: validation accuracy {acc:.4f}", flush=True)
            if acc > best:
                best = acc
                payload = {**run_info, "state_dict": head.state_dict(), "classes": CLASSES,
                           "epoch": epoch, "validation_ids": val.id.tolist()}
                partial = checkpoint.with_suffix(".partial.pt")
                torch.save(payload, partial)
                partial.replace(checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    head.load_state_dict(saved["state_dict"])
    (report / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for name, frame, inputs in (("validation", val, val_x), ("test", test, test_x)):
        probs = predict_head(head, inputs)
        np.savez_compressed(folder / f"{name}_probabilities.npz", ids=frame.id.to_numpy(dtype=str),
                            probabilities=probs, checkpoint_sha256=fingerprint)
        if name == "validation":
            metrics = export_evaluation(frame, probs, report, history)
            metrics["best_epoch"] = saved["epoch"]
            (report / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
