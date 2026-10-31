# Pemilihan convnext_selection

Kandidat terpilih: **equal_architecture_blend**, bobot [0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.3333333333333333].

| Kandidat | Accuracy validasi | Macro-F1 |
| --- | ---: | ---: |
| previous_four_model_blend | 88.2252% | 0.881401 |
| convnext_single | 86.8804% | 0.867754 |
| equal_architecture_blend | 88.3815% | 0.883172 |
| half_convnext_blend | 88.0219% | 0.879404 |

Perubahan terhadap kandidat pertama: +0.16 poin persentase accuracy pada 6,395 halaman validasi yang sama.
Memperbaiki 88 kesalahan dan menambah 78 kesalahan. F1 meningkat pada 9 dari 16 kelas.

Kelas terlemah: scientific_report (0.754), presentation (0.765), form (0.774).

Bobot kandidat ditetapkan sebelum membandingkan hasil validasi.

Holdout yang sama dipakai untuk pemilihan checkpoint dan kandidat. Skor leaderboard yang dilaporkan pengguna adalah 0.88289.

File: `submissions/submission_convnext_selected.csv` (7,994 baris). Hash dan rincian komponen tersedia di `selection.json`.