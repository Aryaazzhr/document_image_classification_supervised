# Pemilihan shift_selection

Kandidat terpilih: **spatial_20_shift**, bobot [0, 0, 0, 0, 0, 0.2, 0.13333333333333333, 0.13333333333333333, 0.13333333333333333, 0.13333333333333333, 0.26666666666666666].

| Kandidat | Accuracy validasi | Macro-F1 |
| --- | ---: | ---: |
| spatial_20_center | 88.4441% | 0.883732 |
| spatial_20_shift | 88.4754% | 0.884134 |

Perubahan terhadap kandidat pertama: +0.03 poin persentase accuracy pada 6,395 halaman validasi yang sama.
Memperbaiki 35 kesalahan dan menambah 33 kesalahan. F1 meningkat pada 8 dari 16 kelas.

Kelas terlemah: scientific_report (0.763), presentation (0.765), form (0.774).

Bobot ensemble sama pada kedua kandidat. Perbedaan skor mengukur dampak rata-rata prediksi lima posisi input.

Holdout yang sama dipakai untuk pemilihan checkpoint dan kandidat. Skor leaderboard menunggu hasil unggah.

File: `submissions/submission_shift_selected.csv` (7,994 baris). Hash dan rincian komponen tersedia di `selection.json`.