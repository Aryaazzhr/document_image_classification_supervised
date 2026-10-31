### Eksperimen `resnet_diversity`

| Metrik | Hasil |
| --- | ---: |
| Accuracy validasi | 86.4113% |
| Macro-F1 validasi | 0.863364 |
| Majority-class baseline accuracy | 6.5676% |
| Halaman training / validasi | 25,607 / 6,395 |
| Epoch terbaik / epoch dijalankan | 9 / 10 |
| Waktu loop training + validasi | 21.0 menit |
| Perangkat | cuda |
| Skor leaderboard | Belum tersedia |

Kelas dengan F1 tertinggi: `email` (0.980), `handwritten` (0.939), `resume` (0.922).
Kelas dengan F1 terendah: `scientific_report` (0.711), `presentation` (0.737), `form` (0.750).

Kesalahan terbanyak (aktual → prediksi): `presentation` → `scientific_report` (35 halaman); `scientific_report` → `presentation` (32 halaman); `news_article` → `presentation` (20 halaman).

Accuracy validasi per epoch berkisar 77.34%–86.41%. Checkpoint dipilih berdasarkan metrik holdout pada konfigurasi. Training loss memakai label smoothing sedangkan validation loss tidak, sehingga nilai absolut keduanya tidak sepenuhnya sebanding.

Hasil ini mengukur satu holdout dengan pemisahan duplikat persis. Holdout juga dipakai untuk memilih checkpoint, sehingga skor dapat optimistis setelah tuning. Skor tersebut bukan public/private leaderboard.

Model `resnet50` menggunakan resolusi 224×224. Kemiripan layout adalah hipotesis penyebab kesalahan, bukan kesimpulan kausal dari confusion matrix. Periksa kelas lemah dan contoh halaman sebelum menentukan eksperimen berikutnya.

Artefak: [metrics](metrics.json), [history](history.csv), [split](split.csv), dan [interpretasi lengkap](interpretation.md).