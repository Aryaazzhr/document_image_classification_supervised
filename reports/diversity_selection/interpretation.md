# Pemilihan diversity_selection

Kandidat terpilih: **diverse_blend**, bobot [0.25, 0.25, 0.5].

| Kandidat | Accuracy validasi | Macro-F1 |
| --- | ---: | ---: |
| previous_ensemble | 86.5207% | 0.864017 |
| resnet_single | 86.4113% | 0.863364 |
| diverse_blend | 87.6310% | 0.875485 |

Perubahan terhadap kandidat pertama: +1.11 poin persentase accuracy pada 6,395 halaman validasi yang sama.
Memperbaiki 164 kesalahan dan menambah 93 kesalahan. F1 meningkat pada 15 dari 16 kelas.

Kelas terlemah: scientific_report (0.733), presentation (0.747), form (0.769).

Hipotesis manfaat diversitas arsitektur dinilai melalui hasil kandidat gabungan. Eksperimen ini tidak memisahkan efek arsitektur, inisialisasi, dan jadwal training secara kausal.

Holdout yang sama sudah digunakan untuk memilih checkpoint dan kandidat. Tidak ada evaluasi CV independen setelah tuning. Tidak ada label test atau lookup duplikat test yang dipakai. Skor leaderboard baru harus dicatat setelah submission.

File: `submissions/submission_diversity_selected.csv` (7,994 baris). Hash dan rincian komponen tersedia di `selection.json`.