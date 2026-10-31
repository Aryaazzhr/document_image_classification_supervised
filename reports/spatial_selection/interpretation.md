# Pemilihan spatial_selection

Kandidat terpilih: **spatial_20**, bobot [0.13333333333333333, 0.13333333333333333, 0.13333333333333333, 0.13333333333333333, 0.26666666666666666, 0.2].

| Kandidat | Accuracy validasi | Macro-F1 |
| --- | ---: | ---: |
| previous | 88.3815% | 0.883172 |
| spatial_only | 82.9554% | 0.828078 |
| spatial_20 | 88.4441% | 0.883732 |
| spatial_40 | 88.3034% | 0.882224 |

Perubahan terhadap kandidat pertama: +0.06 poin persentase accuracy pada 6,395 halaman validasi yang sama.
Memperbaiki 40 kesalahan dan menambah 36 kesalahan. F1 meningkat pada 11 dari 16 kelas.

Kelas terlemah: scientific_report (0.759), presentation (0.768), form (0.769).

Manfaat komponen baru dinilai melalui hasil kandidat tunggal dan gabungan. Eksperimen ini tidak memisahkan efek arsitektur, resolusi, inisialisasi, dan jadwal training secara kausal.

Holdout yang sama sudah digunakan untuk memilih checkpoint dan kandidat. Tidak ada evaluasi CV independen setelah tuning. Tidak ada label test atau lookup duplikat test yang dipakai. Skor leaderboard baru harus dicatat setelah submission.

File: `submissions/submission_spatial_selected.csv` (7,994 baris). Hash dan rincian komponen tersedia di `selection.json`.