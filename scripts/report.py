"""Render measured experiment results into a concise Indonesian interpretation."""

import argparse
import json

import pandas as pd

from scripts.preprocess import CLASSES, ROOT


def write_report(run_name="baseline_cnn"):
    """Summarize actual validation results and refresh the baseline README block."""
    folder = ROOT / "reports" / run_name
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    history = pd.read_csv(folder / "history.csv")
    score_history_path = ROOT / "reports/leaderboard_history.json"
    score_history = json.loads(score_history_path.read_text(encoding="utf-8")) if score_history_path.exists() else []
    scores = [entry for entry in score_history if entry["run_name"] == run_name]
    leaderboard = f"{scores[-1]['score']:.5f} (dilaporkan pengguna; partisi belum disebutkan)" if scores else "Belum tersedia"
    per_class = pd.DataFrame(metrics["classification_report"]).T.loc[CLASSES]
    weakest = per_class.sort_values("f1-score").head(3)
    strongest = per_class.sort_values("f1-score", ascending=False).head(3)
    matrix = pd.read_csv(folder / "confusion_matrix.csv", index_col=0)
    pairs = sorted(((int(matrix.iloc[i, j]), CLASSES[i], CLASSES[j])
                    for i in range(16) for j in range(16) if i != j), reverse=True)[:3]
    lines = [f"### Eksperimen `{run_name}`", "",
             "| Metrik | Hasil |", "| --- | ---: |",
             f"| Accuracy validasi | {metrics['accuracy']:.4%} |",
             f"| Macro-F1 validasi | {metrics['macro_f1']:.6f} |",
             f"| Majority-class baseline accuracy | {metrics['majority_baseline_accuracy']:.4%} |",
             f"| Halaman training / validasi | {run['train_rows']:,} / {run['validation_rows']:,} |",
             f"| Epoch terbaik / epoch dijalankan | {metrics['best_epoch']} / {len(history)} |",
             f"| Waktu loop training + validasi | {history.seconds.sum()/60:.1f} menit |",
             f"| Perangkat | {run['device']} |", f"| Skor leaderboard | {leaderboard} |", "",
             "Kelas dengan F1 tertinggi: " + ", ".join(f"`{name}` ({row['f1-score']:.3f})" for name, row in strongest.iterrows()) + ".",
             "Kelas dengan F1 terendah: " + ", ".join(f"`{name}` ({row['f1-score']:.3f})" for name, row in weakest.iterrows()) + ".", "",
             "Kesalahan terbanyak (aktual → prediksi): " + "; ".join(f"`{actual}` → `{predicted}` ({count} halaman)" for count, actual, predicted in pairs) + ".", "",
             f"Accuracy validasi per epoch berkisar {history.accuracy.min():.2%}–{history.accuracy.max():.2%}. Checkpoint dipilih berdasarkan metrik holdout pada konfigurasi. Training loss memakai label smoothing sedangkan validation loss tidak, sehingga nilai absolut keduanya tidak sepenuhnya sebanding.", "",
             "Hasil ini mengukur satu holdout dengan pemisahan duplikat persis. Holdout juga dipakai untuk memilih checkpoint, sehingga skor dapat optimistis setelah tuning. Skor tersebut bukan public/private leaderboard.", "",
             f"Model `{run['config']['model']}` menggunakan resolusi {run['config']['image_size']}×{run['config']['image_size']}. Kemiripan layout adalah hipotesis penyebab kesalahan, bukan kesimpulan kausal dari confusion matrix. Periksa kelas lemah dan contoh halaman sebelum menentukan eksperimen berikutnya.", "",
             f"Artefak: [metrics](reports/{run_name}/metrics.json), [history](reports/{run_name}/history.csv), [split](reports/{run_name}/split.csv), dan [interpretasi lengkap](reports/{run_name}/interpretation.md)."]
    text = "\n".join(lines)
    (folder / "interpretation.md").write_text(text.replace(f"reports/{run_name}/", ""), encoding="utf-8")
    if run_name == "baseline_cnn":
        readme = ROOT / "README.md"
        content = readme.read_text(encoding="utf-8")
        before, remainder = content.split("<!-- RESULTS_START -->", 1)
        _, after = remainder.split("<!-- RESULTS_END -->", 1)
        readme.write_text(before + "<!-- RESULTS_START -->\n" + text + "\n<!-- RESULTS_END -->" + after, encoding="utf-8")
    print(f"Saved {run_name} interpretation; accuracy={metrics['accuracy']:.6f}, macro_f1={metrics['macro_f1']:.6f}")
    return text


def main():
    """Select the experiment to summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="baseline_cnn")
    write_report(parser.parse_args().run_name)


if __name__ == "__main__":
    main()
