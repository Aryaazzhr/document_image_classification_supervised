### Eksperimen `baseline_cnn`

| Metrik | Hasil |
| --- | ---: |
| Accuracy validasi | 68.0375% |
| Macro-F1 validasi | 0.673017 |
| Majority-class baseline accuracy | 6.5676% |
| Halaman training / validasi | 25,607 / 6,395 |
| Epoch terbaik / epoch dijalankan | 5 / 5 |
| Waktu loop training + validasi | 7.2 menit |
| Perangkat | cpu |
| Skor leaderboard | 0.67294 (dilaporkan pengguna; partisi belum disebutkan) |

Kelas dengan F1 tertinggi: `email` (0.895), `handwritten` (0.861), `file_folder` (0.834).
Kelas dengan F1 terendah: `scientific_report` (0.398), `form` (0.448), `questionnaire` (0.501).

Kesalahan terbanyak (aktual → prediksi): `scientific_report` → `presentation` (56 halaman); `budget` → `invoice` (39 halaman); `presentation` → `scientific_report` (38 halaman).

Accuracy validasi per epoch berkisar 46.04%–68.04%. Fluktuasi ini mendukung pemilihan checkpoint berdasarkan holdout, bukan otomatis memakai epoch terakhir. Training loss memakai label smoothing sedangkan validation loss tidak, sehingga nilai absolut keduanya tidak sepenuhnya sebanding.

Hasil ini mengukur satu holdout dengan pemisahan duplikat persis. Holdout juga dipakai untuk memilih checkpoint, sehingga skor dapat optimistis setelah tuning. Skor tersebut bukan public/private leaderboard.

Model `small_cnn` menggunakan resolusi 128×128. Kemiripan layout adalah hipotesis penyebab kesalahan, bukan kesimpulan kausal dari confusion matrix. Periksa kelas lemah dan contoh halaman sebelum menentukan eksperimen berikutnya.

Artefak: [metrics](metrics.json), [history](history.csv), [split](split.csv), dan [interpretasi lengkap](interpretation.md).