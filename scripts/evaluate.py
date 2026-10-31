"""Export transparent validation metrics, diagnostic figures, and model errors."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from scripts.preprocess import CLASSES


def export_evaluation(frame, probabilities, output_dir, history):
    """Save per-class metrics, confusion matrix, learning curves, and error examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = probabilities.argmax(axis=1)
    target = frame.label.to_numpy()
    metrics = {"accuracy": float(accuracy_score(target, prediction)),
               "macro_f1": float(f1_score(target, prediction, labels=range(16), average="macro", zero_division=0)),
               "validation_rows": len(frame),
               "classification_report": classification_report(target, prediction, labels=range(16),
                                                               target_names=CLASSES, output_dict=True, zero_division=0)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["classification_report"]).T.to_csv(output_dir / "classification_report.csv")
    result = frame.copy()
    result["prediction"] = prediction
    result["predicted_class"] = [CLASSES[i] for i in prediction]
    result["confidence"] = probabilities.max(axis=1)
    result.to_csv(output_dir / "validation_predictions.csv", index=False)
    result.loc[result.label.ne(result.prediction)].sort_values("confidence", ascending=False).head(30).to_csv(
        output_dir / "confident_errors.csv", index=False)
    matrix = confusion_matrix(target, prediction, labels=range(16))
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(output_dir / "confusion_matrix.csv")
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(16), yticks=range(16), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Prediksi", ylabel="Label aktual", title="Confusion matrix (normalisasi per kelas aktual)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    progress = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(progress.epoch, progress.train_loss, label="train")
    axes[0].plot(progress.epoch, progress.val_loss, label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[1].plot(progress.epoch, progress.accuracy, label="accuracy")
    axes[1].plot(progress.epoch, progress.macro_f1, label="macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Skor validasi", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=140)
    plt.close(fig)
    return metrics
