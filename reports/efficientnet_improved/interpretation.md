### Eksperimen `efficientnet_improved`

| Metrik | Hasil |
| --- | ---: |
| Accuracy validasi | 85.4418% |
| Macro-F1 validasi | 0.853316 |
| Majority-class baseline accuracy | 6.5676% |
| Halaman training / validasi | 25,607 / 6,395 |
| Epoch terbaik / epoch dijalankan | 8 / 8 |
| Waktu loop training + validasi | 11.9 menit |
| Perangkat | cuda |
| Skor leaderboard | 0.85236 (dilaporkan pengguna; partisi belum disebutkan) |

Kelas dengan F1 tertinggi: `email` (0.969), `handwritten` (0.934), `resume` (0.929).
Kelas dengan F1 terendah: `scientific_report` (0.689), `form` (0.707), `presentation` (0.725).

Kesalahan terbanyak (aktual → prediksi): `invoice` → `form` (35 halaman); `presentation` → `scientific_report` (31 halaman); `scientific_report` → `presentation` (29 halaman).

Accuracy validasi per epoch berkisar 78.70%–85.44%. Checkpoint dipilih berdasarkan metrik holdout pada konfigurasi. Training loss memakai label smoothing sedangkan validation loss tidak, sehingga nilai absolut keduanya tidak sepenuhnya sebanding.

Hasil ini mengukur satu holdout dengan pemisahan duplikat persis. Holdout juga dipakai untuk memilih checkpoint, sehingga skor dapat optimistis setelah tuning. Skor tersebut bukan public/private leaderboard.

Model `efficientnet_b0` menggunakan resolusi 224×224. Kemiripan layout adalah hipotesis penyebab kesalahan, bukan kesimpulan kausal dari confusion matrix. Periksa kelas lemah dan contoh halaman sebelum menentukan eksperimen berikutnya.

Artefak: [metrics](metrics.json), [history](history.csv), [split](split.csv), dan [interpretasi lengkap](interpretation.md).