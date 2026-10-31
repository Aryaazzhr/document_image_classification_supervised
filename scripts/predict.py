"""Generate an ID-aligned competition submission from a validated checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.model import DocumentDataset, build_model
from scripts.preprocess import CLASSES, ROOT, data_directory, metadata_digest, read_metadata, verified_cache
from scripts.train import make_loader, select_device


def validate_submission(submission, sample):
    """Reject incorrect columns, ID order, missing values, or noninteger labels."""
    if submission.columns.tolist() != ["id", "label"]:
        raise ValueError("Submission columns must be exactly id,label")
    if len(submission) != len(sample) or submission.id.tolist() != sample.id.tolist():
        raise ValueError("Submission IDs must match sample submission order exactly")
    if not submission.id.is_unique or submission.isna().any().any():
        raise ValueError("Submission has duplicate IDs or missing values")
    if not pd.api.types.is_integer_dtype(submission.label) or not submission.label.between(0, 15).all():
        raise ValueError("Submission labels must be integers 0..15")


def generate_submission(checkpoint=None, output=None, data_dir=None, device="auto"):
    """Infer every test image and align predictions by ID with the sample template."""
    checkpoint = Path(checkpoint or ROOT / "checkpoint/baseline_cnn/best.pt")
    output = Path(output or ROOT / "submissions/submission_baseline.csv")
    data_dir = data_directory(data_dir)
    _, test, sample = read_metadata(data_dir)
    selected_device = select_device(device)
    payload = torch.load(checkpoint, map_location=selected_device, weights_only=True)
    if payload["classes"] != CLASSES or payload["metadata_digest"] != metadata_digest(data_dir):
        raise ValueError("Checkpoint class mapping or dataset fingerprint does not match")
    config = payload["config"]
    torch.set_num_threads(config["threads"])
    cache_dir = verified_cache(data_dir, config["image_size"])
    dataset = DocumentDataset(cache_dir / "test.npy", np.arange(len(test)), model_name=config["model"])
    loader = make_loader(dataset, config, False, selected_device)
    model = build_model(config["model"], pretrained=False).to(selected_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    probabilities = np.empty((len(test), 16), dtype=np.float32)
    with torch.inference_mode():
        for images, indices in loader:
            probabilities[indices.numpy()] = model(images.to(selected_device)).softmax(dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    np.savez_compressed(checkpoint.parent / "test_probabilities.npz", ids=test.id.to_numpy(dtype=str),
                        probabilities=probabilities,
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    by_id = pd.Series(predictions, index=test.id)
    submission = pd.DataFrame({"id": sample.id, "label": sample.id.map(by_id).astype("int64")})
    validate_submission(submission, sample)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    validate_submission(pd.read_csv(output), sample)
    provenance = {"checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "submission_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "rows": len(submission), "label_counts": submission.label.value_counts().sort_index().to_dict(),
                  "best_epoch": payload["epoch"], "validation_metrics": payload["metrics"],
                  "public_score": None, "private_score": None}
    (ROOT / "reports" / f"{output.stem}_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Validated submission: {output} ({len(submission)} rows)", flush=True)
    return submission


def main():
    """Parse inference arguments and write a schema-checked submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoint/baseline_cnn/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "submissions/submission_baseline.csv")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    generate_submission(args.checkpoint, args.output, args.data_dir, args.device)


if __name__ == "__main__":
    main()
