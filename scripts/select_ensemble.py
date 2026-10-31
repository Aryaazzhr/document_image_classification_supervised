"""Compare prespecified ensembles across architectures on an identical holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch

from scripts.predict import validate_submission
from scripts.preprocess import CLASSES, ROOT, data_directory, metadata_digest, read_metadata
from scripts.select_highres import aligned_probabilities
from scripts.train import validate_initial_checkpoint


def validate_candidates(candidates, count):
    """Reject ambiguous names or weights that do not define a probability mixture."""
    names = [item["variant"] for item in candidates]
    if not names or len(set(names)) != len(names):
        raise ValueError("Candidates need distinct names")
    for item in candidates:
        weights = np.asarray(item["weights"], dtype=float)
        if weights.shape != (count,) or not np.isfinite(weights).all():
            raise ValueError("Each candidate needs one finite weight per component")
        if (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError("Mixture weights must be nonnegative and sum to one")


def load_component(run_name, frame, split, test_ids, digest, view=None):
    """Check split, checkpoint, and prediction identity before adding an ensemble member."""
    folder = ROOT / "checkpoint" / run_name
    report = ROOT / "reports" / run_name
    checkpoint = folder / "best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = json.loads((report / "run.json").read_text(encoding="utf-8"))["config"]
    validate_initial_checkpoint(payload, config["model"], digest, frame.id.tolist())
    current_split = pd.read_csv(report / "split.csv").sort_values("id").reset_index(drop=True)
    if not split.equals(current_split):
        raise ValueError(f"Training/validation split differs: {run_name}")
    if view is not None:
        if view != "shift4":
            raise ValueError(f"Unknown inference view: {view}")
        folder, report = folder / view, report / view
        manifest = json.loads((report / "inference.json").read_text())
        if (manifest["checkpoint_sha256"] != checkpoint_hash or manifest["metadata_digest"] != digest
                or manifest["view"] != view):
            raise ValueError("Inference views do not match the checkpoint and dataset")
    with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as archive:
        has_hash = "checkpoint_sha256" in archive
    if view and not has_hash:
        raise ValueError("Shifted inference must identify the checkpoint")
    validation = aligned_probabilities(folder / "validation_probabilities.npz", frame.id.tolist(),
                                       checkpoint_hash if has_hash else None)
    recorded = pd.read_csv(report / "validation_predictions.csv").set_index("id").loc[frame.id]
    if not np.array_equal(recorded.label.to_numpy(), frame.label.to_numpy()):
        raise ValueError(f"Validation labels differ: {run_name}")
    if not np.array_equal(recorded.prediction.to_numpy(), validation.argmax(axis=1)):
        raise ValueError(f"Validation archive differs from recorded predictions: {run_name}")
    testing = aligned_probabilities(folder / "test_probabilities.npz", test_ids, checkpoint_hash)
    component = {"run_name": run_name, "model": config["model"],
                 "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": checkpoint_hash,
                 "validation_archive_sha256": hashlib.sha256((folder / "validation_probabilities.npz").read_bytes()).hexdigest(),
                 "validation_identity": "checkpoint_hash" if has_hash else "legacy_archive_with_matching_recorded_predictions"}
    if "image_size" in config:
        component["image_size"] = config["image_size"]
    else:
        component["page_wh"] = config["page_wh"]
        component["header_wh"] = config["header_wh"]
        component["source_run"] = config["source"]
        source = ROOT / "checkpoint" / config["source"] / "best.pt"
        component["source_checkpoint_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        if payload["source_checkpoint_sha256"] != component["source_checkpoint_sha256"]:
            raise ValueError(f"Feature extractor has changed: {run_name}")
    if view:
        component["inference_view"] = manifest
    return validation, testing, component


def select_ensemble(config_path):
    """Select a fixed candidate by accuracy, export diagnostics, and generate aligned test labels."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = config["runs"]
    validate_candidates(config["candidates"], len(runs))
    data_dir = data_directory()
    _, test, sample = read_metadata(data_dir)
    reference = ROOT / "reports" / runs[0]
    frame = pd.read_csv(reference / "validation_predictions.csv")
    split = pd.read_csv(reference / "split.csv").sort_values("id").reset_index(drop=True)
    views = config.get("views", [None] * len(runs))
    loaded = [load_component(name, frame, split, test.id.tolist(), metadata_digest(data_dir), view)
              for name, view in zip(runs, views, strict=True)]
    validation = np.stack([item[0] for item in loaded])
    testing = np.stack([item[1] for item in loaded])
    records, best, selected_probabilities = [], None, None
    for candidate in config["candidates"]:
        probabilities = np.tensordot(candidate["weights"], validation, axes=1)
        prediction = probabilities.argmax(axis=1)
        record = {**candidate, "accuracy": float(accuracy_score(frame.label, prediction)),
                  "macro_f1": float(f1_score(frame.label, prediction, labels=range(16), average="macro", zero_division=0))}
        records.append(record)
        if best is None or record["accuracy"] > best["accuracy"]:
            best, selected_probabilities = record, probabilities
    predicted_test = np.tensordot(best["weights"], testing, axes=1).argmax(axis=1)
    by_id = pd.Series(predicted_test, index=test.id)
    submission = pd.DataFrame({"id": sample.id, "label": sample.id.map(by_id).astype("int64")})
    validate_submission(submission, sample)
    output = ROOT / config["submission"]
    output.parent.mkdir(exist_ok=True)
    if output.exists() and not pd.read_csv(output).equals(submission):
        raise FileExistsError("Preserve the existing submission by choosing a different output path")
    submission.to_csv(output, index=False)
    validate_submission(pd.read_csv(output), sample)
    report = ROOT / "reports" / config["experiment"]
    report.mkdir(exist_ok=True)
    predicted = selected_probabilities.argmax(axis=1)
    old_predicted = np.tensordot(config["candidates"][0]["weights"], validation, axes=1).argmax(axis=1)
    old_correct, new_correct = old_predicted == frame.label.to_numpy(), predicted == frame.label.to_numpy()
    summary = {"config": config, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
               "candidates": records, "selected": best, "components": [item[2] for item in loaded],
               "validation_rows": len(frame), "test_rows": len(submission),
               "corrected_errors": int((~old_correct & new_correct).sum()),
               "new_errors": int((old_correct & ~new_correct).sum()),
               "submission": output.relative_to(ROOT).as_posix(),
               "submission_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "leaderboard_score": None,
               "limitation": "Shared holdout reused for checkpoint and candidate selection; no independent CV estimate"}
    history = json.loads((ROOT / "reports/leaderboard_history.json").read_text(encoding="utf-8"))
    matches = [item for item in history if item["submission_sha256"] == summary["submission_sha256"]]
    if matches:
        summary["leaderboard_score"] = matches[-1]["score"]
        summary["score_source"] = matches[-1]["score_source"]
        summary["reported_on"] = matches[-1]["reported_on"]
    (report / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(report / "candidates.csv", index=False)
    class_report = classification_report(frame.label, predicted, labels=range(16), target_names=CLASSES,
                                         output_dict=True, zero_division=0)
    pd.DataFrame(class_report).T.to_csv(report / "classification_report.csv")
    previous_report = classification_report(frame.label, old_predicted, labels=range(16),
                                            target_names=CLASSES, output_dict=True, zero_division=0)
    class_comparison = pd.DataFrame({"class_name": CLASSES,
                                     "previous_f1": [previous_report[name]["f1-score"] for name in CLASSES],
                                     "selected_f1": [class_report[name]["f1-score"] for name in CLASSES]})
    class_comparison["delta_f1"] = class_comparison.selected_f1 - class_comparison.previous_f1
    class_comparison.to_csv(report / "class_comparison.csv", index=False)
    pd.DataFrame(confusion_matrix(frame.label, predicted, labels=range(16)), index=CLASSES,
                 columns=CLASSES).to_csv(report / "confusion_matrix.csv")
    frame["prediction"], frame["predicted_class"] = predicted, [CLASSES[label] for label in predicted]
    frame["confidence"] = selected_probabilities.max(axis=1)
    frame.to_csv(report / "validation_predictions.csv", index=False)
    frame.loc[frame.label.ne(frame.prediction)].sort_values("confidence", ascending=False).head(20).to_csv(
        report / "confident_errors.csv", index=False)
    write_interpretation(report, summary, class_comparison)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def write_interpretation(report, summary, class_comparison):
    """Explain the measured paired change and remaining weak classes without claiming causation."""
    selected, previous = summary["selected"], summary["candidates"][0]
    weakest = class_comparison.sort_values("selected_f1").head(3)
    method = ("Bobot ensemble sama pada kedua kandidat. Perbedaan skor mengukur dampak rata-rata prediksi lima posisi input."
              if summary["config"].get("views") else "Bobot kandidat ditetapkan sebelum membandingkan hasil validasi.")
    score_note = ("Skor leaderboard menunggu hasil unggah."
                  if summary["leaderboard_score"] is None else f"Skor leaderboard yang dilaporkan pengguna adalah {summary['leaderboard_score']:.5f}.")
    text = [f"# Pemilihan {summary['config']['experiment']}", "",
            f"Kandidat terpilih: **{selected['variant']}**, bobot {selected['weights']}.", "",
            "| Kandidat | Accuracy validasi | Macro-F1 |", "| --- | ---: | ---: |"]
    text.extend(f"| {row['variant']} | {row['accuracy']:.4%} | {row['macro_f1']:.6f} |"
                for row in summary["candidates"])
    text.extend(["", f"Perubahan terhadap kandidat pertama: {100*(selected['accuracy']-previous['accuracy']):+.2f} poin persentase accuracy pada {summary['validation_rows']:,} halaman validasi yang sama.",
                 f"Memperbaiki {summary['corrected_errors']} kesalahan dan menambah {summary['new_errors']} kesalahan. F1 meningkat pada {int(class_comparison.delta_f1.gt(0).sum())} dari 16 kelas.", "",
                 "Kelas terlemah: " + ", ".join(f"{row.class_name} ({row.selected_f1:.3f})" for row in weakest.itertuples()) + ".", "",
                 method, "",
                 "Holdout yang sama dipakai untuk pemilihan checkpoint dan kandidat. " + score_note, "",
                 f"File: `{summary['submission']}` ({summary['test_rows']:,} baris). Hash dan rincian komponen tersedia di `selection.json`."])
    (report / "interpretation.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diversity_selection.json")
    select_ensemble(parser.parse_args().config)
