"""Compare two experiments on exactly matching validation IDs and source labels."""

import argparse
import json

import pandas as pd

from scripts.preprocess import CLASSES, ROOT


def compare_runs(candidate="efficientnet_improved", reference="baseline_cnn"):
    """Validate split equivalence and export aggregate, per-class, and paired changes."""
    base = ROOT / "reports" / reference
    improved = ROOT / "reports" / candidate
    old_split = pd.read_csv(base / "split.csv").sort_values("id").reset_index(drop=True)
    new_split = pd.read_csv(improved / "split.csv").sort_values("id").reset_index(drop=True)
    if not old_split.equals(new_split):
        raise ValueError("Experiments used different splits; their scores are not directly comparable")
    old = pd.read_csv(base / "validation_predictions.csv").set_index("id").sort_index()
    new = pd.read_csv(improved / "validation_predictions.csv").set_index("id").sort_index()
    if not old.index.equals(new.index) or not old.label.equals(new.label):
        raise ValueError("Validation IDs or ground-truth labels do not match")
    old_metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    new_metrics = json.loads((improved / "metrics.json").read_text(encoding="utf-8"))
    table = pd.DataFrame([
        {"run": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
         "validation_rows": metrics["validation_rows"], "best_epoch": metrics["best_epoch"]}
        for name, metrics in ((reference, old_metrics), (candidate, new_metrics))
    ])
    table.to_csv(improved / "comparison.csv", index=False)
    per_class = pd.DataFrame({"class_name": CLASSES,
                              "baseline_f1": [old_metrics["classification_report"][c]["f1-score"] for c in CLASSES],
                              "candidate_f1": [new_metrics["classification_report"][c]["f1-score"] for c in CLASSES]})
    per_class["delta_f1"] = per_class.candidate_f1 - per_class.baseline_f1
    per_class.to_csv(improved / "class_comparison.csv", index=False)
    old_correct, new_correct = old.label.eq(old.prediction), new.label.eq(new.prediction)
    summary = {"reference": reference, "candidate": candidate, "identical_split": True,
               "accuracy_gain_percentage_points": 100 * (new_metrics["accuracy"] - old_metrics["accuracy"]),
               "macro_f1_gain": new_metrics["macro_f1"] - old_metrics["macro_f1"],
               "corrected_errors": int((~old_correct & new_correct).sum()),
               "new_errors": int((old_correct & ~new_correct).sum()),
               "both_correct": int((old_correct & new_correct).sum()),
               "both_wrong": int((~old_correct & ~new_correct).sum()),
               "classes_with_f1_improvement": int(per_class.delta_f1.gt(0).sum()),
               "candidate_leaderboard_score": reported_score(candidate)}
    summary["recommended_run"] = candidate if new_metrics["accuracy"] > old_metrics["accuracy"] else reference
    (improved / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_readme(reference, candidate, old_metrics, new_metrics, summary)
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


def reported_score(run_name):
    """Retrieve the latest user-reported score without inferring its leaderboard partition."""
    path = ROOT / "reports/leaderboard_history.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    matches = [row for row in history if row["run_name"] == run_name]
    return matches[-1]["score"] if matches else None


def update_readme(reference, candidate, old_metrics, new_metrics, summary):
    """Write a run-specific comparison using actual settings and recorded leaderboard scores."""
    path = ROOT / "README.md"
    content = path.read_text(encoding="utf-8")
    marker = "IMPROVEMENT" if candidate == "efficientnet_improved" else candidate.upper()
    start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
    if start not in content or end not in content:
        content = content.replace("## Struktur repository", f"{start}\n{end}\n\n## Struktur repository", 1)
    run = json.loads((ROOT / "reports" / candidate / "run.json").read_text(encoding="utf-8"))
    config = run["config"]
    old_score, new_score = reported_score(reference), reported_score(candidate)
    old_score_text = f"{old_score:.5f} (laporan pengguna)" if old_score is not None else "Belum tersedia"
    new_score_text = f"{new_score:.5f} (laporan pengguna)" if new_score is not None else "Belum tersedia"
    initialization = (f"Checkpoint `{config['init_checkpoint']}`" if config.get("init_checkpoint")
                      else "Bobot ImageNet" if config["pretrained"] else "Bobot acak")
    lines = [f"## Perbandingan `{candidate}`", "",
             "Skor leaderboard berasal dari laporan pengguna. Jenis public/private dan nama metrik belum disebutkan.", "",
             "| Eksperimen | Accuracy holdout | Macro-F1 holdout | Skor leaderboard |",
             "| --- | ---: | ---: | ---: |",
             f"| {reference} | {old_metrics['accuracy']:.4%} | {old_metrics['macro_f1']:.6f} | {old_score_text} |",
             f"| {candidate} | {new_metrics['accuracy']:.4%} | {new_metrics['macro_f1']:.6f} | {new_score_text} |", "",
             f"Peningkatan accuracy: **{summary['accuracy_gain_percentage_points']:.2f} poin persentase**. Kedua run memakai ID training/validasi dan label yang identik. Kandidat memperbaiki {summary['corrected_errors']:,} kesalahan baseline, tetapi membuat {summary['new_errors']:,} kesalahan baru.", "",
             f"F1 meningkat pada **{summary['classes_with_f1_improvement']} dari 16 kelas**. Nilai per kelas tersedia pada laporan perbandingan.", "",
             f"{initialization} digunakan untuk inisialisasi `{config['model']}` pada {config['image_size']}×{config['image_size']}. Optimizer AdamW, LR awal {config['learning_rate']}, cosine schedule, maksimum {config['epochs']} epoch, batch {config['batch_size']}, dan label smoothing {config['label_smoothing']}. Checkpoint dipilih berdasarkan {config['selection_metric']} validasi.", "",
             "Beberapa pengaturan berubah bersama; eksperimen ini belum memisahkan kontribusi masing-masing. Holdout yang sama dipakai untuk pemilihan checkpoint, sehingga kenaikan skor lokal belum merupakan jaminan kenaikan leaderboard. Tidak ada lookup label dari duplikat test.", "",
             f"Run yang direkomendasikan berdasarkan accuracy holdout: **`{summary['recommended_run']}`**.", "",
             f"Lihat [perbandingan](reports/{candidate}/comparison.json), [per kelas](reports/{candidate}/class_comparison.csv), [interpretasi](reports/{candidate}/interpretation.md), dan [riwayat leaderboard](reports/leaderboard_history.json)."]
    before, remainder = content.split(start, 1)
    _, after = remainder.split(end, 1)
    path.write_text(before + start + "\n" + "\n".join(lines) + "\n" + end + after, encoding="utf-8")


def main():
    """Select reference and candidate experiment names from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="efficientnet_improved")
    parser.add_argument("--reference", default="baseline_cnn")
    args = parser.parse_args()
    compare_runs(args.candidate, args.reference)


if __name__ == "__main__":
    main()
