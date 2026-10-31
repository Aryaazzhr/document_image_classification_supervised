### Eksperimen `resnet_highres`

| Metrik | Hasil |
| --- | ---: |
| Accuracy validasi | 86.6145% |
| Macro-F1 validasi | 0.864731 |
| Majority-class baseline accuracy | 6.5676% |
| Halaman training / validasi | 25,607 / 6,395 |
| Epoch terbaik / epoch dijalankan | 3 / 6 |
| Waktu loop training + validasi | 26.0 menit |
| Perangkat | cuda |
| Skor leaderboard | Belum tersedia |

Kelas dengan F1 tertinggi: `email` (0.975), `handwritten` (0.936), `file_folder` (0.931).
Kelas dengan F1 terendah: `scientific_report` (0.714), `presentation` (0.729), `form` (0.735).

Kesalahan terbanyak (aktual → prediksi): `scientific_report` → `presentation` (34 halaman); `presentation` → `scientific_report` (29 halaman); `form` → `invoice` (22 halaman).

Accuracy validasi per epoch berkisar 85.63%–86.61%. Checkpoint dipilih berdasarkan metrik holdout pada konfigurasi. Training loss memakai label smoothing sedangkan validation loss tidak, sehingga nilai absolut keduanya tidak sepenuhnya sebanding.

Hasil ini mengukur satu holdout dengan pemisahan duplikat persis. Holdout juga dipakai untuk memilih checkpoint, sehingga skor dapat optimistis setelah tuning. Skor tersebut bukan public/private leaderboard.

Model `resnet50` menggunakan resolusi 320×320. Kemiripan layout adalah hipotesis penyebab kesalahan, bukan kesimpulan kausal dari confusion matrix. Periksa kelas lemah dan contoh halaman sebelum menentukan eksperimen berikutnya.

Artefak: [metrics](metrics.json), [history](history.csv), [split](split.csv), dan [interpretasi lengkap](interpretation.md).