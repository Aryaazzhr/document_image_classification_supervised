### Eksperimen `convnext_diversity`

| Metrik | Hasil |
| --- | ---: |
| Accuracy validasi | 86.8804% |
| Macro-F1 validasi | 0.867754 |
| Majority-class baseline accuracy | 6.5676% |
| Halaman training / validasi | 25,607 / 6,395 |
| Epoch terbaik / epoch dijalankan | 9 / 10 |
| Waktu loop training + validasi | 24.8 menit |
| Perangkat | cuda |
| Skor leaderboard | Belum tersedia |

Kelas dengan F1 tertinggi: `email` (0.968), `handwritten` (0.932), `file_folder` (0.926).
Kelas dengan F1 terendah: `scientific_report` (0.722), `form` (0.743), `presentation` (0.750).

Kesalahan terbanyak (aktual → prediksi): `scientific_report` → `presentation` (33 halaman); `presentation` → `scientific_report` (27 halaman); `scientific_report` → `scientific_publication` (23 halaman).

Accuracy validasi per epoch berkisar 78.26%–86.88%. Checkpoint dipilih berdasarkan metrik holdout pada konfigurasi. Training loss memakai label smoothing sedangkan validation loss tidak, sehingga nilai absolut keduanya tidak sepenuhnya sebanding.

Hasil ini mengukur satu holdout dengan pemisahan duplikat persis. Holdout juga dipakai untuk memilih checkpoint, sehingga skor dapat optimistis setelah tuning. Skor tersebut bukan public/private leaderboard.

Model `convnext_tiny` menggunakan resolusi 224×224. Kemiripan layout adalah hipotesis penyebab kesalahan, bukan kesimpulan kausal dari confusion matrix. Periksa kelas lemah dan contoh halaman sebelum menentukan eksperimen berikutnya.

Artefak: [metrics](metrics.json), [history](history.csv), [split](split.csv), dan [interpretasi lengkap](interpretation.md).