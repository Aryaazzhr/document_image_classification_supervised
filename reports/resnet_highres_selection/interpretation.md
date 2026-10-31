# Pemilihan resnet_highres_selection

Kandidat terpilih: **four_model_blend**, bobot [0.25, 0.25, 0.25, 0.25].

| Kandidat | Accuracy validasi | Macro-F1 |
| --- | ---: | ---: |
| previous_diverse_blend | 87.6310% | 0.875485 |
| resnet_highres_single | 86.6145% | 0.864731 |
| replace_resnet_224 | 88.0688% | 0.879886 |
| four_model_blend | 88.2252% | 0.881401 |

Perubahan terhadap kandidat pertama: +0.59 poin persentase accuracy pada 6,395 halaman validasi yang sama.
Memperbaiki 78 kesalahan dan menambah 40 kesalahan. F1 meningkat pada 13 dari 16 kelas.

Kelas terlemah: scientific_report (0.749), presentation (0.754), form (0.767).

Manfaat komponen baru dinilai melalui hasil kandidat tunggal dan gabungan. Eksperimen ini tidak memisahkan efek arsitektur, resolusi, inisialisasi, dan jadwal training secara kausal.

Holdout yang sama sudah digunakan untuk memilih checkpoint dan kandidat. Tidak ada evaluasi CV independen setelah tuning. Tidak ada label test atau lookup duplikat test yang dipakai. Skor leaderboard baru harus dicatat setelah submission.

File: `submissions/submission_resnet_highres_selected.csv` (7,994 baris). Hash dan rincian komponen tersedia di `selection.json`.