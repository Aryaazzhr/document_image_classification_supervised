### Eksperimen `efficientnet_highres`

| Metrik | Hasil |
| --- | ---: |
| Accuracy validasi | 86.2392% |
| Macro-F1 validasi | 0.861266 |
| Majority-class baseline accuracy | 6.5676% |
| Halaman training / validasi | 25,607 / 6,395 |
| Epoch terbaik / epoch dijalankan | 5 / 6 |
| Waktu loop training + validasi | 16.2 menit |
| Perangkat | cuda |
| Skor leaderboard | Belum tersedia |

Kelas dengan F1 tertinggi: `email` (0.978), `handwritten` (0.935), `file_folder` (0.924).
Kelas dengan F1 terendah: `scientific_report` (0.702), `form` (0.736), `presentation` (0.738).

Kesalahan terbanyak (aktual → prediksi): `scientific_report` → `presentation` (32 halaman); `presentation` → `scientific_report` (31 halaman); `budget` → `scientific_report` (23 halaman).

Accuracy validasi per epoch berkisar 85.02%–86.24%. Checkpoint dipilih berdasarkan metrik holdout pada konfigurasi. Training loss memakai label smoothing sedangkan validation loss tidak, sehingga nilai absolut keduanya tidak sepenuhnya sebanding.

Hasil ini mengukur satu holdout dengan pemisahan duplikat persis. Holdout juga dipakai untuk memilih checkpoint, sehingga skor dapat optimistis setelah tuning. Skor tersebut bukan public/private leaderboard.

Model `efficientnet_b0` menggunakan resolusi 320×320. Kemiripan layout adalah hipotesis penyebab kesalahan, bukan kesimpulan kausal dari confusion matrix. Periksa kelas lemah dan contoh halaman sebelum menentukan eksperimen berikutnya.

Artefak: [metrics](metrics.json), [history](history.csv), [split](split.csv), dan [interpretasi lengkap](interpretation.md).