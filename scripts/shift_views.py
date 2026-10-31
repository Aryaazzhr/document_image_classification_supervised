"""Average center and four small translations for each previously trained CNN."""

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.model import build_model
from scripts.preprocess import CLASSES, ROOT, read_metadata, data_directory, verified_cache, metadata_digest
from scripts.select_highres import aligned_probabilities
from scripts.train import validate_initial_checkpoint


def shifted_probabilities(model, pages, indices, center, size, device):
    """Average five probabilities with identical weights and white-filled border pixels."""
    distance = round(size * 0.014)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    result = np.empty_like(center, dtype=np.float32)
    batch_size = 16 if size >= 320 else 24
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            subset = indices[start:start + batch_size]
            raw = torch.from_numpy(pages[subset]).to(device)[:, None]
            padded = F.pad(raw, (distance,) * 4, value=255)
            views = torch.cat([padded[:, :, y:y+size, x:x+size]
                               for y in (0, 2 * distance) for x in (0, 2 * distance)])
            images = (views.float().div_(255) - mean) / std
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
            shifted = logits.float().softmax(1).reshape(4, len(subset), 16).sum(0).cpu().numpy()
            result[start:start + len(subset)] = (shifted + center[start:start + len(subset)]) / 5
    return result


def run():
    """Export ID-aligned validation and test probabilities for the shifted ensemble."""
    config = json.loads((ROOT / "configs/shift_selection.json").read_text())
    runs = [name for name, view in zip(config["runs"], config["views"], strict=True) if view == "shift4"]
    data_dir = data_directory()
    train, test, _ = read_metadata(data_dir)
    digest = metadata_digest(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    for name in runs:
        folder = ROOT / "checkpoint" / name
        report = ROOT / "reports" / name
        output = folder / "shift4"
        destination = report / "shift4"
        output.mkdir(exist_ok=True)
        destination.mkdir(exist_ok=True)
        val = pd.read_csv(report / "validation_predictions.csv")
        checkpoint_hash = hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest()
        payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        setting = payload["config"]
        validate_initial_checkpoint(payload, setting["model"], digest, val.id.tolist())
        size = setting["image_size"]
        manifest = {"view": "shift4", "metadata_digest": digest, "checkpoint_sha256": checkpoint_hash,
                    "distance_px": round(0.014 * size), "padding": "white", "device": device.type,
                    "recipe": "original center FP32 plus four diagonal translations with equal weights",
                    "shifted_precision": "float16 autocast" if device.type == "cuda" else "float32"}
        marker = destination / "inference.json"
        required = [output / f"{part}_probabilities.npz" for part in ("validation", "test")]
        if (marker.exists() and json.loads(marker.read_text()) == manifest and all(p.exists() for p in required)
                and (destination / "validation_predictions.csv").exists()):
            continue
        marker.unlink(missing_ok=True)
        model = build_model(setting["model"]).to(device).eval()
        model.load_state_dict(payload["state_dict"])
        del payload
        cache = verified_cache(data_dir, size)
        indices = pd.Index(train.id).get_indexer(val.id)
        if (indices < 0).any() or not np.array_equal(train.label.to_numpy()[indices], val.label):
            raise ValueError("Invalid validation metadata alignment")
        for part, frame, rows, source in (("validation", val, indices, "train"),
                                         ("test", test, np.arange(len(test)), "test")):
            path = folder / f"{part}_probabilities.npz"
            with np.load(path, allow_pickle=False) as archive:
                check_hash = checkpoint_hash if "checkpoint_sha256" in archive or part == "test" else None
            center = aligned_probabilities(path, frame.id.tolist(), check_hash)
            if part == "validation" and not np.array_equal(center.argmax(1), frame.prediction):
                raise ValueError("Center archive differs from recorded validation predictions")
            pages = np.load(cache / f"{source}.npy", mmap_mode="r")
            probs = shifted_probabilities(model, pages, rows, center, size, device)
            target = output / f"{part}_probabilities.npz"
            partial = target.with_suffix(".partial.npz")
            np.savez_compressed(partial, ids=frame.id.to_numpy(dtype=str), probabilities=probs,
                                checkpoint_sha256=checkpoint_hash)
            partial.replace(target)
            if part == "validation":
                frame = frame.copy()
                frame["prediction"] = probs.argmax(1)
                frame["predicted_class"] = [CLASSES[i] for i in frame.prediction]
                frame["confidence"] = probs.max(1)
                frame.to_csv(destination / "validation_predictions.csv", index=False)
            print(f"{name} {part}: {len(frame)} views averaged", flush=True)
        marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        del model


if __name__ == "__main__":
    run()
