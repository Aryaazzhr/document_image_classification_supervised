"""Select the high-resolution model or a fixed equal-weight blend on a shared holdout."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch

from scripts.predict import validate_submission
from scripts.preprocess import CLASSES, ROOT, data_directory, metadata_digest, read_metadata
from scripts.train import validate_initial_checkpoint


def aligned_probabilities(path, expected_ids, checkpoint_hash=None):
    """Validate a prediction archive and align its probabilities by unique sample ID."""
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str)
        probabilities = archive["probabilities"].astype(np.float64)
        if checkpoint_hash is not None:
            if "checkpoint_sha256" not in archive or str(archive["checkpoint_sha256"].item()) != checkpoint_hash:
                raise ValueError("Test probability cache does not match its checkpoint")
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("Prediction IDs differ from the expected evaluation set")
    if probabilities.shape != (len(ids), 16) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite probabilities of shape (n, 16)")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability distributions")
    return probabilities[pd.Index(ids).get_indexer(expected_ids)]


def select_highres():
    """Compare two individual models and one predeclared 50/50 blend, then write a submission."""
    data_dir = data_directory()
    _, test, sample = read_metadata(data_dir)
    runs = ["efficientnet_improved", "efficientnet_highres"]
    frame = pd.read_csv(ROOT / "reports/efficientnet_highres/validation_predictions.csv")
    validation, testing, components = [], [], []
    reference_split = pd.read_csv(ROOT / "reports/efficientnet_improved/split.csv").sort_values("id").reset_index(drop=True)
    for run_name in runs:
        folder = ROOT / "checkpoint" / run_name
        checkpoint_path = folder / "best.pt"
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_initial_checkpoint(payload, "efficientnet_b0", metadata_digest(data_dir), frame.id.tolist())
        current_split = pd.read_csv(ROOT / "reports" / run_name / "split.csv").sort_values("id").reset_index(drop=True)
        if not reference_split.equals(current_split):
            raise ValueError("Component training/validation splits differ")
        probabilities = aligned_probabilities(folder / "validation_probabilities.npz", frame.id.tolist())
        recorded = pd.read_csv(ROOT / "reports" / run_name / "validation_predictions.csv").set_index("id").loc[frame.id]
        if not np.array_equal(recorded.label.to_numpy(), frame.label.to_numpy()):
            raise ValueError("Component validation labels differ")
        if not np.array_equal(probabilities.argmax(axis=1), recorded.prediction.to_numpy()):
            raise ValueError("Validation archive disagrees with the recorded checkpoint predictions")
        validation.append(probabilities)
        testing.append(aligned_probabilities(folder / "test_probabilities.npz", test.id.tolist(), checkpoint_hash))
        components.append({"run_name": run_name, "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                           "checkpoint_sha256": checkpoint_hash})
    candidates = [("highres_single", [0.0, 1.0]), ("previous_single", [1.0, 0.0]),
                  ("equal_blend", [0.5, 0.5])]
    records, best_score, selected, selected_probabilities = [], -1.0, None, None
    for name, weights in candidates:
        probabilities = weights[0] * validation[0] + weights[1] * validation[1]
        predictions = probabilities.argmax(axis=1)
        accuracy = float(accuracy_score(frame.label, predictions))
        record = {"variant": name, "weights": weights, "accuracy": accuracy,
                  "macro_f1": float(f1_score(frame.label, predictions, labels=range(16), average="macro", zero_division=0))}
        records.append(record)
        if accuracy > best_score:
            best_score, selected, selected_probabilities = accuracy, record, probabilities
    selected_test = selected["weights"][0] * testing[0] + selected["weights"][1] * testing[1]
    predicted_by_id = pd.Series(selected_test.argmax(axis=1), index=test.id)
    submission = pd.DataFrame({"id": sample.id, "label": sample.id.map(predicted_by_id).astype("int64")})
    validate_submission(submission, sample)
    output = ROOT / "submissions/submission_highres_selected.csv"
    submission.to_csv(output, index=False)
    validate_submission(pd.read_csv(output), sample)
    report = ROOT / "reports/highres_selection"
    report.mkdir(exist_ok=True)
    summary = {"selection_rule": "Highest holdout accuracy; ties prefer the highres single model",
               "candidates": records, "selected": selected, "validation_rows": len(frame),
               "components": components, "submission": str(output.relative_to(ROOT)),
               "submission_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "test_rows": len(submission), "leaderboard_score": None,
               "limitation": "The shared holdout was used for both checkpoint and blend selection; no independent CV estimate"}
    history_path = ROOT / "reports/leaderboard_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        matches = [item for item in history if item["submission_sha256"] == summary["submission_sha256"]]
        if matches:
            summary["leaderboard_score"] = matches[-1]["score"]
            summary["reported_on"] = matches[-1]["reported_on"]
            summary["score_source"] = matches[-1]["score_source"]
    (report / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(report / "candidates.csv", index=False)
    predictions = selected_probabilities.argmax(axis=1)
    class_report = classification_report(frame.label, predictions, labels=range(16), target_names=CLASSES,
                                         output_dict=True, zero_division=0)
    (report / "classification_report.json").write_text(json.dumps(class_report, indent=2), encoding="utf-8")
    pd.DataFrame(class_report).T.to_csv(report / "classification_report.csv")
    pd.DataFrame(confusion_matrix(frame.label, predictions, labels=range(16)), index=CLASSES,
                 columns=CLASSES).to_csv(report / "confusion_matrix.csv")
    frame["prediction"] = predictions
    frame["predicted_class"] = [CLASSES[label] for label in predictions]
    frame["confidence"] = selected_probabilities.max(axis=1)
    frame.to_csv(report / "validation_predictions.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    select_highres()
