# Klasifikasi dokumen bisnis dari hasil scan

Proyek praktikum Supervised Machine Learning ini mengelompokkan halaman TIFF ke dalam 16 jenis dokumen. Model akhir menggabungkan prediksi visual dari EfficientNet, ResNet, dan ConvNeXt dengan kata-kata yang terbaca pada bagian atas halaman. Skornya **0,90046 di leaderboard Kaggle**. [Riwayat submission](reports/leaderboard_history.json)

## Hasil eksperimen

| Model | Akurasi validasi | Macro-F1 | Skor leaderboard |
| --- | ---: | ---: | ---: |
| CNN awal | 68,04% | 0,6730 | 0,67294 |
| EfficientNet-B0 pada 224 × 224 | 85,44% | 0,8533 | 0,85236 |
| EfficientNet-B0 pada dua resolusi | 86,52% | 0,8640 | 0,86491 |
| Ensemble dengan ConvNeXt | 88,38% | 0,8832 | 0,88289 |
| Ensemble visual dengan fitur layout dan translasi | 88,48% | 0,8841 | — |
| Teks header saja | 64,85% | 0,6500 | — |
| **Ensemble visual dan teks header** | **89,48%** | **0,8939** | **0,90046** |

Semua model dievaluasi pada 6.395 halaman yang sama. Untuk tahap terakhir, Windows OCR membaca 45% bagian atas tiap scan. Classifier teks mendapat bobot 38% jika sedikitnya 12 kata terbaca. Untuk header yang lebih pendek, bobotnya turun sebanding dengan jumlah kata. Ada 1.305 halaman validasi dengan bobot OCR nol.

Penambahan teks memperbaiki 113 kesalahan model visual dan menambah 49 kesalahan baru. Selisihnya 64 halaman, atau 1,00 poin persentase akurasi. F1 `resume` naik dari 0,935 menjadi 0,960, sedangkan F1 `scientific_publication` turun dari 0,925 menjadi 0,913. Interval bootstrap berpasangan 95% untuk kenaikan akurasi adalah +0,62 sampai +1,39 poin pada holdout ini. Holdout yang sama juga dipakai saat memilih model visual. [Perubahan per kelas](reports/ocr_header/class_comparison.csv) · [Notebook evaluasi](notebooks/10_ocr_header.ipynb)

Tahap akhir menghasilkan `submissions/submission_ocr_header.csv` dengan 7.994 baris dalam urutan ID Kaggle. Notebook menyimpan hasil eksekusi agar grafik dan evaluasi bisa dibaca langsung di GitHub. Untuk melatih ulang model, unduh dataset dan ikuti perintah di bawah. [Konfigurasi dan hash submission](reports/ocr_header/result.json)

<details>
<summary>Laporan training tiap model</summary>

<!-- RESULTS_START -->
- [baseline_cnn](reports/baseline_cnn/interpretation.md)
<!-- RESULTS_END -->
<!-- IMPROVEMENT_START -->
- [efficientnet_improved](reports/efficientnet_improved/interpretation.md)
<!-- IMPROVEMENT_END -->
<!-- EFFICIENTNET_HIGHRES_START -->
- [efficientnet_highres](reports/efficientnet_highres/interpretation.md)
<!-- EFFICIENTNET_HIGHRES_END -->
<!-- RESNET_DIVERSITY_START -->
- [resnet_diversity](reports/resnet_diversity/interpretation.md)
<!-- RESNET_DIVERSITY_END -->
<!-- RESNET_HIGHRES_START -->
- [resnet_highres](reports/resnet_highres/interpretation.md)
<!-- RESNET_HIGHRES_END -->

</details>

## Struktur repository

```text
document-image-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-tested.txt   # versi environment CPU yang telah diuji
├── requirements-gpu.txt      # pasangan PyTorch/torchvision CUDA
├── configs/
│   ├── baseline.json
│   ├── efficientnet_improved.json
│   ├── efficientnet_highres.json
│   ├── resnet_diversity.json
│   ├── diversity_selection.json
│   ├── resnet_highres.json
│   ├── resnet_highres_selection.json
│   ├── convnext_diversity.json
│   ├── convnext_selection.json
│   ├── spatial_layout.json
│   ├── spatial_selection.json
│   ├── shift_selection.json
│   ├── ocr_header.json
│   └── efficientnet.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_submission.ipynb
│   ├── 04_improve.ipynb
│   ├── 05_highres.ipynb
│   ├── 06_diversity.ipynb
│   ├── 07_resnet_highres.ipynb
│   ├── 08_convnext.ipynb
│   ├── 09_spatial_layout.ipynb
│   └── 10_ocr_header.ipynb
├── scripts/
│   ├── preprocess.py          # audit, cache, split, visualisasi EDA
│   ├── model.py               # Dataset dan arsitektur PyTorch
│   ├── train.py               # training, checkpoint selection
│   ├── evaluate.py            # metrik, confusion matrix, error analysis
│   ├── predict.py             # inference dan validasi submission
│   ├── report.py              # ringkasan hasil aktual
│   ├── compare_runs.py        # verifikasi split dan perbandingan kandidat
│   ├── select_highres.py      # seleksi model atau ensemble 50:50
│   ├── select_ensemble.py     # kandidat ensemble lintas arsitektur
│   ├── spatial_layout.py      # fitur posisi dan header, serta classifier PyTorch
│   ├── read_header_ocr.ps1    # pembacaan TIFF dan OCR lokal Windows
│   ├── extract_ocr.py         # cache teks per ID dan pemeriksaan urutan
│   ├── ocr_classifier.py      # classifier PyTorch dan evaluasi gabungan
│   ├── shift_views.py         # rata-rata probabilitas lima posisi halaman
│   ├── check_gpu.py           # uji langkah training CUDA
│   ├── build_notebooks.py     # generator notebook tanpa output
│   ├── execute_notebooks.py   # eksekusi dan simpan output aktual
│   └── validate_repository.py # pemeriksaan docstring dan artefak
├── tests/
│   └── test_pipeline.py
├── reports/                  # laporan kecil yang dapat di-commit
│   ├── data_audit.json
│   ├── image_errors.json
│   ├── figures/
│   └── baseline_cnn/
├── data/                     # diabaikan Git: dataset/cache
├── checkpoint/               # diabaikan Git: bobot model
└── submissions/              # diabaikan Git: hasil prediksi lokal
```

## Cara memperoleh data

1. Buka [Document Image Classification — SD234504](https://www.kaggle.com/competitions/document-image-classification-sd234504/data), login dengan akun yang memiliki akses kelas, dan setujui aturan kompetisi.
2. Dari tab **Data**, unduh `train.zip`, `test.zip`, `train.csv`, `test.csv`, dan `sample_submission.csv`.
3. Simpan CSV ke `data/`, lalu ekstrak arsip sehingga tersedia `data/train/<class>/<file>.tif` dan `data/test/<file>.tif`. Jangan menggabungkan dataset dari kompetisi lain.
4. Loader juga menerima ekstraksi bertingkat `data/train/train/` dan `data/test/test/`. Jika metadata berada di tempat lain, gunakan `--data-dir "path-ke-folder-csv"` pada preprocess, train, dan predict.
5. Jalankan audit sebelum training. Jangan commit dataset atau arsip ke GitHub.

Contoh ekstraksi PowerShell, dari root proyek setelah file unduhan diletakkan di `data/`:

```powershell
Expand-Archive -LiteralPath data/train.zip -DestinationPath data
Expand-Archive -LiteralPath data/test.zip -DestinationPath data
```

Loader mencari `train.csv` di `data/`, lalu di root proyek untuk struktur ekstraksi lama.

## Instalasi

Gunakan Python 3.11 atau lebih baru dengan pasangan PyTorch/torchvision yang kompatibel. Dari root proyek:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Untuk GPU, instal pasangan PyTorch dan torchvision memakai perintah untuk Windows/CUDA dari [pemilih instalasi resmi PyTorch](https://pytorch.org/get-started/locally/), lalu pasang requirements lainnya. Driver NVIDIA saja tidak membuat paket PyTorch CPU dapat menggunakan GPU. `--device cuda` akan berhenti dengan pesan jelas jika CUDA tidak tersedia.

Environment eksekusi awal menggunakan Python 3.14 dan PyTorch 2.12.0+cpu dengan torchvision 0.27.0+cpu. `.venv` lokal memanfaatkan paket sistem yang sudah ada, ditambah dependency notebook dan torchvision yang kompatibel; environment tersebut bukan artefak untuk di-commit. Versi aktual eksperimen disimpan di `reports/<run>/run.json` dan paket pengujian dicatat pada `requirements-tested.txt`. Snapshot CPU ini bukan konfigurasi instalasi CUDA.

## Jalankan pipeline baseline

```powershell
python -m scripts.preprocess --image-size 128
python -m scripts.train --config configs/baseline.json
python -m scripts.predict --checkpoint checkpoint/baseline_cnn/best.pt --output submissions/submission_baseline.csv
python -m scripts.report --run-name baseline_cnn
```

Preprocessing pertama membaca seluruh 39.997 gambar dan memerlukan waktu. Cache grayscale uint8 untuk 128×128 memakai sekitar 655 MB, lalu training membaca cache melalui memory map. Cache 320×320 memakai sekitar 4,1 GB. Cache dipisahkan berdasarkan resolusi dan dicek terhadap fingerprint CSV. Jika gambar atau metadata berubah, jalankan preprocessing ulang; perubahan gambar tidak dideteksi hanya dari fingerprint CSV.

Training tidak menimpa checkpoint yang sudah ada. Gunakan nama run baru untuk mengulang atau mengubah eksperimen:

```powershell
python -m scripts.train --config configs/baseline.json --run-name baseline_repeat
```

Output per run: konfigurasi dan versi paket, daftar ID split, history per epoch, `best.pt`, accuracy, macro-F1, precision/recall/F1 per kelas, confusion matrix, learning curves, dan prediksi salah dengan confidence tinggi. Checkpoint ditujukan untuk inferensi; resume optimizer belum diimplementasikan.

## Notebook sesuai workflow praktikum

```powershell
python -m jupyterlab
```

Jalankan notebook secara berurutan:

1. **01_eda** — problem understanding, metadata, audit gambar, distribusi kelas, sampel visual, keputusan preprocessing.
2. **02_modeling** — konfigurasi, split, training PyTorch, evaluasi, dan interpretasi kesalahan.
3. **03_submission** — inference, validasi ID/label, provenance, dan pencatatan leaderboard.

Notebook membaca artefak yang tersedia; jika belum ada, tahap terkait dijalankan. Untuk mengeksekusi dan menyimpan seluruh output secara otomatis:

```powershell
python -m scripts.execute_notebooks
```

`build_notebooks` hanya dipakai untuk membuat ulang template dan **menghapus output notebook lama**. Tidak perlu menjalankannya sebelum memakai notebook yang sudah disediakan.

## Data understanding dan pencegahan leakage

- Metadata: 32.003 training, 7.994 test, 16 kelas, tanpa nilai kosong atau ID/filename duplikat.
- Kelas berkisar 1.948–2.056 halaman; rasio terbesar/terkecil sekitar 1,055. Baseline tidak memakai class weighting.
- Satu TIFF rusak: `train_08795`, `train/scientific_publication/2500126531_2500126536.tif`. File sumber dipertahankan; row dikeluarkan dari training **dan** validasi. Ada **32.002 halaman layak pakai**.
- Ditemukan 48 baris training duplikat berdasarkan hash piksel grayscale asli beserta dimensi, serta 3 grup duplikat dengan label konflik. Label asli dipertahankan dan konflik dilaporkan. Hash dihitung sebelum resize.
- Ada 25 gambar test dengan piksel yang sama dengan gambar training. Temuan ini hanya dicatat sebagai audit; seluruh prediksi test tetap berasal dari jaringan, tanpa lookup label berdasarkan hash.
- Semua gambar test terbaca. Tidak ada TIFF multipage pada audit ini; jika ditemukan kemudian, hanya halaman pertama digunakan.

Split menggunakan `StratifiedGroupKFold` dengan 5 fold, shuffle, seed 42, dan fold pertama sebagai holdout sekitar 20%. Gambar identik tidak melintasi split. **Ini bukan 5-fold cross-validation**: hanya satu fold yang dievaluasi. ID grup dokumen sumber tidak tersedia, sehingga hubungan antarhalaman dan near-duplicates belum seluruhnya ditangani. Audit dilakukan sebelum split tanpa mengubah label berdasarkan test.

## Preprocessing dan model

Pada cabang visual, TIFF dibaca dengan Pillow, orientasi EXIF diterapkan jika tersedia, kemudian halaman dikonversi ke grayscale. Halaman di-resize dengan rasio aspek tetap dan padding putih menjadi persegi. Training mendapat perubahan kontras ringan. Run highres dan ResNet juga memakai affine ringan yang dijelaskan di bawah. Tahap OCR membaca kata langsung dari piksel pada header TIFF asli.

| Komponen | CNN baseline | EfficientNet-B0 (`efficientnet_improved`) |
| --- | --- | --- |
| Input | 1 × 128 × 128 | 3 × 224 × 224, grayscale direplikasi |
| Inisialisasi | Acak | ImageNet pretrained |
| Arsitektur | Conv 16/32/64/96, BatchNorm, ReLU; grid 4×4; FC 256 → 16 | Backbone torchvision; head diganti menjadi 16 output |
| Optimizer | AdamW, LR 0,001 | AdamW, LR 0,0003 |
| Batch / epoch maksimum | 64 / 5 | 32 / 8 |
| Regularisasi | Dropout 0,3; weight decay 0,0001 | Dropout bawaan; weight decay 0,01 |
| Normalisasi | Mean/std 0,5 | Mean/std ImageNet |

Keduanya memakai cross entropy dengan label smoothing 0,05, cosine scheduler, gradient clipping 1,0, dan early stopping. Checkpoint terbaik dipilih dengan **accuracy validasi** secara default. Accuracy dan macro-F1 sama-sama dilaporkan karena spesifikasi belum menyebut metrik kompetisi. Ganti `selection_metric` menjadi `macro_f1` bila aturan resmi mengharuskannya, lalu buat run baru.

Torchvision menyediakan bobot ImageNet untuk [EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0). Pipeline memakai padding halaman pada ukuran input konfigurasi agar tepi halaman tetap tersedia. Peningkatan skor harus dibuktikan pada split yang sama; konfigurasi pretrained bukan bukti hasil training.

## Fine-tuning EfficientNet di GPU

Eksperimen lanjutan menggunakan environment `.venv-gpu` terpisah dengan pasangan CUDA pada `requirements-gpu.txt`. Environment CPU baseline tetap tersedia. Contoh instalasi baru:

```powershell
python -m venv .venv-gpu
.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Reproduksi kandidat 224×224:

```powershell
.venv-gpu\Scripts\python.exe -m scripts.preprocess --image-size 224
.venv-gpu\Scripts\python.exe -m scripts.train --config configs/efficientnet_improved.json --device cuda
.venv-gpu\Scripts\python.exe -m scripts.predict --checkpoint checkpoint/efficientnet_improved/best.pt --output submissions/submission_improved.csv --device cuda
.venv-gpu\Scripts\python.exe -m scripts.compare_runs
.venv-gpu\Scripts\python.exe -m scripts.report --run-name efficientnet_improved
```

Cache 224×224 memerlukan sekitar 2,0 GB. Notebook `04_improve.ipynb` memperlihatkan perbandingan pada ID validasi yang sama. Konfigurasi 320×320 di bawah tetap tersedia sebagai eksperimen tambahan, bukan run yang otomatis dijalankan:

```powershell
python -m scripts.preprocess --image-size 320
python -m scripts.train --config configs/efficientnet.json --device cuda
python -m scripts.predict --checkpoint checkpoint/efficientnet_b0/best.pt --output submissions/submission_efficientnet.csv --device cuda
python -m scripts.report --run-name efficientnet_b0
```

Unduhan bobot pertama disimpan di `checkpoint/pretrained/`. Semua layer di-fine-tune. Mixed precision aktif hanya pada CUDA. Turunkan `batch_size` jika memori GPU tidak cukup; gunakan run baru dan simpan konfigurasi. CNN baseline dapat berjalan tanpa torchvision; EfficientNet memerlukannya.

## Kelanjutan fine-tuning 320×320

Konfigurasi `efficientnet_highres.json` memakai checkpoint `efficientnet_improved/best.pt`, learning rate 0,00006, batch 24, maksimal 6 epoch tambahan, dan optimizer AdamW baru. Pipeline memverifikasi fingerprint dataset, class mapping, arsitektur, serta ID holdout sebelum memuat bobot. Ini melanjutkan bobot model tanpa memulihkan state optimizer lama.

Augmentasi training menambahkan rotasi ±1,5°, translasi ±1,5%, dan skala 0,97–1,03 dengan area luar berwarna putih. Validasi serta inference tetap deterministik. Cache dibuat dari TIFF asli pada 320×320, bukan memperbesar cache 224×224. Resolusi, augmentasi, dan jadwal training berubah bersama; eksperimen ini tidak mengisolasi efek masing-masing.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.preprocess --image-size 320
.venv-gpu\Scripts\python.exe -m scripts.train --config configs/efficientnet_highres.json --device cuda
.venv-gpu\Scripts\python.exe -m scripts.predict --checkpoint checkpoint/efficientnet_highres/best.pt --output submissions/submission_highres.csv --device cuda
.venv-gpu\Scripts\python.exe -m scripts.predict --checkpoint checkpoint/efficientnet_improved/best.pt --output submissions/submission_improved_recheck.csv --device cuda
.venv-gpu\Scripts\python.exe -m scripts.compare_runs --candidate efficientnet_highres --reference efficientnet_improved
.venv-gpu\Scripts\python.exe -m scripts.report --run-name efficientnet_highres
.venv-gpu\Scripts\python.exe -m scripts.select_highres
```

Perintah inference kedua membuat cache probabilitas model 224 tanpa menimpa submission yang sudah mendapat skor. `select_highres` membandingkan tiga kandidat yang ditetapkan sebelum hasil akhir: model 224, model 320, dan rata-rata probabilitas 50:50. Tidak ada pencarian bobot per kelas. ID disejajarkan secara eksplisit dan hash checkpoint pada cache test diverifikasi. Kandidat dipilih berdasarkan accuracy holdout; jika seri, model tunggal 320 diprioritaskan.

File rekomendasi tahap ini adalah **`submissions/submission_highres_selected.csv`**. Komponen, bobot, metrik validasi, dan hash file dicatat di `reports/highres_selection/selection.json`. Pemakaian holdout untuk pemilihan model dan ensemble dapat membuat estimasi lokal optimistis; skor leaderboard baru tetap perlu diperiksa. Lihat `notebooks/05_highres.ipynb` untuk output eksekusi dan analisis.

## Diversitas arsitektur: ResNet-50

Eksperimen `resnet_diversity` menggunakan [ResNet-50 ImageNet-1K V2 dari Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html). Head diganti menjadi 16 kelas dan semua layer di-fine-tune. Input 224×224, batch 32, AdamW LR 0,0002, weight decay 0,01, label smoothing 0,05, cosine schedule 10 epoch, patience 4. Augmentasi geometri sama dengan tahap highres. Cache 224 yang sudah ada digunakan kembali.

Hipotesisnya adalah kesalahan backbone baru dapat melengkapi dua EfficientNet yang saling terkait melalui warm start. Kenaikan skor tidak diasumsikan dari ukuran model. Tiga kandidat ditetapkan pada `configs/diversity_selection.json` sebelum hasil training: ensemble lama, ResNet tunggal, dan rata-rata probabilitas ensemble lama dengan ResNet. Bobot akhir kandidat gabungan adalah 0,25 / 0,25 / 0,50 untuk EfficientNet 224 / EfficientNet 320 / ResNet 224. Accuracy tertinggi dipilih; seri memilih kandidat paling awal, sehingga ensemble lama dipertahankan jika hasil setara.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.train --config configs/resnet_diversity.json --device cuda
.venv-gpu\Scripts\python.exe -m scripts.predict --checkpoint checkpoint/resnet_diversity/best.pt --output submissions/submission_resnet.csv --device cuda
.venv-gpu\Scripts\python.exe -m scripts.report --run-name resnet_diversity
.venv-gpu\Scripts\python.exe -m scripts.select_ensemble --config configs/diversity_selection.json
.venv\Scripts\python.exe -m scripts.execute_notebooks --notebook 06_diversity.ipynb
```

Selesaikan dua eksperimen EfficientNet dan inference sebelumnya terlebih dahulu. Selector memeriksa split lengkap, ID, label mapping, fingerprint data, dan hash checkpoint pada cache test. Arsip validasi ResNet juga menyimpan hash checkpoint. Arsip EfficientNet lama diperiksa terhadap prediksi validasi yang tersimpan, dengan keterbatasan provenance ini dicatat pada laporan. Seluruh kandidat menggunakan holdout yang sama; ini tetap bukan evaluasi independen setelah tuning berulang.

Output: `submissions/submission_diversity_selected.csv`, laporan pemilihan `reports/diversity_selection/selection.json`, dan notebook `06_diversity.ipynb`. Tidak ada lookup label test, tuning bobot per kelas, atau pengubahan submission yang telah mendapat skor.

## ResNet 320×320: kelanjutan fine-tuning

Konfigurasi `resnet_highres.json` melanjutkan checkpoint terbaik ResNet 224 (epoch 9) pada cache TIFF 320 dengan LR 0,00003, batch 24, maksimum 6 epoch, dan patience 3. Optimizer dimulai ulang. Augmentasi, label smoothing, weight decay, dan split dipertahankan. Cache 320 yang telah dibuat untuk EfficientNet digunakan kembali; tidak perlu menyimpan salinan dataset tambahan.

Empat kandidat ditetapkan pada `configs/resnet_highres_selection.json`: ensemble sebelumnya, ResNet 320 tunggal, ensemble yang mengganti ResNet 224 dengan ResNet 320, serta rata-rata empat model. Urutan komponen adalah EfficientNet 224 / EfficientNet 320 / ResNet 224 / ResNet 320. Pilih accuracy tertinggi; seri memilih kandidat paling awal. Tidak ada pencarian bobot per kelas.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.train --config configs/resnet_highres.json --device cuda
.venv-gpu\Scripts\python.exe -m scripts.predict --checkpoint checkpoint/resnet_highres/best.pt --output submissions/submission_resnet_highres.csv --device cuda
.venv-gpu\Scripts\python.exe -m scripts.compare_runs --candidate resnet_highres --reference resnet_diversity
.venv-gpu\Scripts\python.exe -m scripts.report --run-name resnet_highres
.venv-gpu\Scripts\python.exe -m scripts.select_ensemble --config configs/resnet_highres_selection.json
.venv\Scripts\python.exe -m scripts.execute_notebooks --notebook 07_resnet_highres.ipynb
```

Prasyarat: semua checkpoint dan cache probabilitas tahap sebelumnya tersedia. Hasil pemilihan dicatat pada `reports/resnet_highres_selection/selection.json`; outputnya `submissions/submission_resnet_highres_selected.csv`. Notebook `07_resnet_highres.ipynb` memeriksa hash submission lama dan baru. Pemakaian holdout berulang untuk checkpoint dan kandidat dapat membuat estimasi lokal optimistis; skor leaderboard baru belum diketahui.

## ConvNeXt-Tiny: arsitektur tambahan

Eksperimen `convnext_diversity` menggunakan [bobot ImageNet-1K V1 resmi Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html). Head diubah menjadi 16 kelas; semua layer di-fine-tune pada training split yang sama. Input letterbox 224×224, normalisasi ImageNet, batch 24, AdamW LR 0,0001, weight decay 0,05, label smoothing 0,05, cosine schedule maksimum 10 epoch, patience 4. Stochastic depth mengikuti default torchvision 0,1. Augmentasi affine/kontras sama dengan eksperimen ResNet; tidak ada OCR atau lookup label test.

Kandidat tetap pada `configs/convnext_selection.json`: ensemble empat model sebelumnya, ConvNeXt tunggal, gabungan dengan total bobot 1/3 per keluarga arsitektur, dan gabungan 50:50 ensemble lama dengan ConvNeXt. Kandidat ditetapkan sebelum training; accuracy tertinggi dipilih, dan seri mempertahankan kandidat paling awal. Tidak ada tuning bobot per kelas.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.train --config configs/convnext_diversity.json --device cuda
.venv-gpu\Scripts\python.exe -m scripts.predict --checkpoint checkpoint/convnext_diversity/best.pt --output submissions/submission_convnext.csv --device cuda
.venv-gpu\Scripts\python.exe -m scripts.report --run-name convnext_diversity
.venv-gpu\Scripts\python.exe -m scripts.select_ensemble --config configs/convnext_selection.json
.venv\Scripts\python.exe -m scripts.execute_notebooks --notebook 08_convnext.ipynb
```

Tahap ini menggunakan checkpoint dan probabilitas empat model sebelumnya. Skor leaderboard untuk `submissions/submission_convnext_selected.csv` adalah 0,88289. Rincian komponennya tercatat pada `reports/convnext_selection/selection.json`.

## Fitur layout halaman dan header

Pipeline membaca TIFF asli dalam dua tampilan. Tampilan pertama memuat seluruh halaman dalam bidang berukuran 320×416 piksel (lebar × tinggi). Tampilan kedua memuat 40% bagian atas halaman dalam bidang 384×192 piksel. Keduanya memakai padding putih untuk menjaga rasio aspek. Checkpoint ConvNeXt 224 dari tahap sebelumnya dibekukan selama ekstraksi fitur.

Posisi fitur dipertahankan melalui pooling per wilayah. Untuk halaman utuh, grid pada dua tahap terakhir ConvNeXt berukuran 3×3 dan 2×2. Untuk header, grid berukuran 2×3 dan 1×1. Classifier PyTorch dengan 384 unit tersembunyi menerima 9.600 fitur per halaman. Training memakai AdamW, batch 256, LR awal 0,0007, dan cosine schedule selama 16 epoch. ID training dan validasi mengikuti split checkpoint sumber. Cache fitur disimpan di `checkpoint/spatial_layout/` dan membutuhkan sekitar 0,77 GB.

Mulai dari TIFF asli, checkpoint ConvNeXt, dan probabilitas ensemble tahap sebelumnya. Jalankan perintah berikut dari root proyek.

```powershell
.venv-gpu\Scripts\python.exe -m scripts.spatial_layout
.venv\Scripts\python.exe -m scripts.select_ensemble --config configs/spatial_selection.json
.venv-gpu\Scripts\python.exe -m scripts.shift_views
.venv\Scripts\python.exe -m scripts.select_ensemble --config configs/shift_selection.json
.venv\Scripts\python.exe -m scripts.execute_notebooks --notebook 09_spatial_layout.ipynb
```

Setelah tahap layout, uji translasi mempertahankan bobot kandidat layout 20%. Untuk lima CNN, prediksi tengah dirata-ratakan dengan empat prediksi dari pergeseran diagonal. Jarak pergeseran adalah 3 piksel pada input 224 dan 4 piksel pada input 320. Batas halaman diisi putih dan setiap posisi mendapat bobot ⅕. Prediksi layout berbobot 20% memakai tampilan halaman dan header aslinya. Pembanding memakai posisi tengah dengan bobot ensemble identik. Tahap ini memerlukan empat inferensi ekstra per CNN per halaman.

## Eksperimen teks header

OCR lokal Windows membaca 45% bagian atas setiap halaman dengan faktor pembesaran 1,35 dan paket pengenalan bahasa Inggris `en-US`. Prasyarat ekstraksi adalah Windows dengan Windows PowerShell 5.1 dan paket OCR tersebut. Masukan classifier hanya isi gambar. ID dipakai untuk menyelaraskan baris, dan digit pada hasil pembacaan dinormalisasi agar nomor seri tidak menjadi kata unik. [Windows OCR API](https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr.ocrengine)

Teks diproses dengan TF-IDF pada kata (1–2 gram) dan potongan kata (3–5 karakter). Masing-masing kosakata dipelajari dari 25.607 halaman training. Jaringan PyTorch menjumlahkan embedding dengan bobot TF-IDF kemudian memakai classifier dengan 128 unit tersembunyi. Jumlah epoch tetap 12. Konfigurasi bobot OCR ditetapkan sebelum melihat akurasi. Pada halaman dengan setidaknya 12 kata alfabetis, bobotnya 38%. Untuk header lebih pendek bobotnya proporsional dengan jumlah kata, hingga 0% untuk header kosong.

Mulai dari TIFF, metadata, dan hasil ensemble `shift_selection` yang sudah direproduksi. Ekstraksi menyimpan batch yang sudah lolos pemeriksaan ID di `data/ocr_cache/`. Cache teks, kosakata, dan bobot model berada di direktori yang diabaikan Git.

```powershell
.venv\Scripts\python.exe -u -m scripts.extract_ocr
.venv-gpu\Scripts\python.exe -u -m scripts.ocr_classifier
.venv\Scripts\python.exe -u -m scripts.execute_notebooks --notebook 10_ocr_header.ipynb
```

Satu kandidat dibandingkan dengan model visual pada ID yang sama. Syarat rekomendasi adalah kenaikan akurasi dan macro-F1 serta batas bawah interval bootstrap berpasangan 95% untuk selisih akurasi di atas nol. Bootstrap mengulang sampling 2.000 kali dengan halaman duplikat persis tetap satu grup. Interval ini menggambarkan ketidakpastian pada holdout yang telah dipakai dalam beberapa eksperimen. Keputusan, metrik, dan hash file yang dipilih tersedia di `reports/ocr_header/result.json`.

## Submission dan leaderboard

Submission wajib memiliki kolom `id,label`, tepat satu baris per ID test, tanpa NaN, dan label integer 0–15. Prediksi dipetakan dengan ID dan diurutkan mengikuti template, sehingga tidak bergantung pada urutan direktori. File hasil dibaca ulang untuk memastikan formatnya valid. Hash checkpoint dan submission disimpan pada `reports/*_provenance.json`.

Model baseline yang disubmit adalah checkpoint holdout; belum dilakukan refit seluruh training. Label contoh pada `sample_submission.csv` tidak pernah digunakan untuk melatih atau menilai model. Upload file melalui halaman kompetisi secara manual, lalu catat public/private score resmi. Bobot nilai tugas: public 10%, private 20%, kualitas repository 70%.

## Pemeriksaan sebelum GitHub

```powershell
python -m unittest discover -s tests -v
python -m scripts.validate_repository
git status --short
```

Pengujian mencakup pembacaan TIFF bertingkat, padding tanpa crop, pemisahan duplikat, pengecualian gambar rusak, kontrak submission, satu langkah optimizer nyata, dan docstring semua function/class `.py`. Validator memeriksa notebook, source, dan ukuran artefak yang ditujukan untuk commit. Periksa juga `git diff --cached --stat` sebelum commit: `.gitignore` tidak menghapus file besar yang telanjur tracked.
