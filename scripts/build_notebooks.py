"""Build narrative notebooks using the reference practical's structure."""

import json
from pathlib import Path

from scripts.preprocess import ROOT


def cell(kind, source):
    """Create a notebook cell with readable source lines and no fabricated outputs."""
    item = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def write_notebook(name, contents):
    """Save a valid version-4 notebook that can be executed in Jupyter or nbclient."""
    notebook = {"cells": [cell(kind, source) for kind, source in contents],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.14.0"}},
                "nbformat": 4, "nbformat_minor": 4}
    target = ROOT / "notebooks" / name
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def build_highres_notebook():
    """Create a separate notebook documenting checkpoint continuation and blend selection."""
    write_notebook("05_highres.ipynb", [
        ("markdown", """# 05 — Fine-tuning 320×320 dan pemilihan submission
Skor leaderboard model 224×224 yang dilaporkan pengguna pada 28 Oktober 2026: **0,85236**. Eksperimen ini melanjutkan checkpoint tersebut pada resolusi lebih tinggi, dengan learning rate rendah dan augmentasi geometri ringan. Split training/validasi dipertahankan.

Tiga kandidat ditetapkan sebelum melihat hasil akhir: model 224, model 320, dan rata-rata probabilitas 50:50. Pemilihan berdasarkan accuracy holdout. Skor ini bukan estimasi CV independen karena holdout juga dipakai memilih checkpoint."""),
        ("code", """from pathlib import Path
import sys, json, hashlib
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'reports/efficientnet_highres'
display(json.loads((ROOT / 'configs/efficientnet_highres.json').read_text()))
display(pd.read_json(ROOT / 'reports/leaderboard_history.json'))"""),
        ("markdown", """## Inisialisasi dan pencegahan leakage
Checkpoint sumber sudah dilatih pada training split yang sama. Pipeline menolak inisialisasi jika fingerprint data, arsitektur, class mapping, atau ID validasinya berbeda. Optimizer dan cosine schedule dimulai ulang; ini adalah tahap fine-tuning baru, bukan resume optimizer.

Halaman asli di-resize ke 320×320 dengan padding putih. Augmentasi training: rotasi hingga 1,5°, translasi hingga 1,5%, skala 0,97–1,03, serta kontras ringan. Tidak ada flip. Validasi dan inference tidak memakai augmentasi acak. Resolusi, jadwal optimizer, dan augmentasi berubah bersama, sehingga efek masing-masing belum diisolasi."""),
        ("code", """assert (RUN / 'metrics.json').exists(), 'Jalankan configs/efficientnet_highres.json terlebih dahulu'
from scripts.compare_runs import compare_runs
comparison = compare_runs(candidate='efficientnet_highres', reference='efficientnet_improved')
display(json.loads((RUN / 'run.json').read_text()))
display(pd.read_csv(RUN / 'comparison.csv'))
display(pd.read_csv(RUN / 'history.csv'))"""),
        ("markdown", "## Evaluasi model 320×320\nBandingkan accuracy, macro-F1, dan perubahan F1 per kelas. Periksa kelas yang masih sering tertukar, bukan hanya rata-rata keseluruhan."),
        ("code", """display(Image(filename=str(RUN / 'learning_curves.png')))
display(Image(filename=str(RUN / 'confusion_matrix.png')))
display(pd.read_csv(RUN / 'class_comparison.csv').sort_values('delta_f1'))
display(pd.read_csv(RUN / 'confident_errors.csv').head(10))"""),
        ("markdown", """## Kombinasi dua resolusi
Probabilitas disejajarkan berdasarkan ID. Hash checkpoint pada cache test diverifikasi. Kombinasi 50:50 dibandingkan dengan kedua model tunggal; jika accuracy seri, model tunggal 320 diprioritaskan. Tidak ada pencarian bobot per kelas, penggunaan label test, atau lookup label dari gambar duplikat.

Pemilihan ini menambah penggunaan holdout untuk tuning. Hasil leaderboard harus diperiksa lewat submission baru."""),
        ("code", """SELECTED = ROOT / 'reports/highres_selection'
selection = json.loads((SELECTED / 'selection.json').read_text())
display(pd.read_csv(SELECTED / 'candidates.csv'))
display(selection)
display(pd.read_csv(SELECTED / 'classification_report.csv', index_col=0))"""),
        ("markdown", "## Validasi submission terpilih\nSubmission yang dipakai adalah `submission_highres_selected.csv`. Dua submission sebelumnya dipertahankan bersama riwayat skor dan hash file masing-masing."),
        ("code", """from scripts.preprocess import read_metadata, data_directory
from scripts.predict import validate_submission
output = ROOT / selection['submission']
submission = pd.read_csv(output)
_, test, sample = read_metadata(data_directory())
validate_submission(submission, sample)
assert len(submission) == len(test)
assert hashlib.sha256(output.read_bytes()).hexdigest() == selection['submission_sha256']
print(f'Valid: {len(submission)} rows; variant={selection["selected"]["variant"]}')
display(submission.head())"""),
    ])


def build_diversity_notebook():
    """Document architecture diversity, fixed ensemble candidates, and submission checks."""
    write_notebook("06_diversity.ipynb", [
        ("markdown", """# 06 — ResNet-50 dan ensemble lintas arsitektur
Ensemble EfficientNet 224/320 sebelumnya mendapat skor leaderboard **0,86491**, dilaporkan pengguna pada 28 Oktober 2026. Kelas lemah pada validasi mencakup scientific_report, presentation, dan form. Hipotesis eksperimen: fitur dari backbone berbeda dapat mengurangi kesalahan bersama kedua EfficientNet.

ResNet-50 dimulai dari [bobot ImageNet-1K V2 resmi Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), bukan dari checkpoint kompetisi eksternal. Model hanya menerima piksel. Split yang sama dipakai untuk semua komponen; duplikat persis tidak melintasi train/validasi."""),
        ("code", """from pathlib import Path
import sys, json, hashlib
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'reports/resnet_diversity'
SELECTED = ROOT / 'reports/diversity_selection'
display(pd.read_json(ROOT / 'reports/leaderboard_history.json'))
display(pd.read_csv(ROOT / 'reports/highres_selection/classification_report.csv', index_col=0).iloc[:16].sort_values('f1-score').head())"""),
        ("markdown", """## Preprocessing dan training
Gambar letterbox 224×224 mempertahankan rasio halaman dan padding putih. Grayscale direplikasi menjadi tiga kanal lalu dinormalisasi ImageNet. Augmentasi training: kontras, rotasi maksimum 1,5°, translasi maksimum 1,5%, dan skala 0,97–1,03; tanpa flip. Validasi tidak memakai augmentasi acak.

Semua layer di-fine-tune dengan AdamW, LR 0,0002, cosine schedule, label smoothing 0,05, batch 32, maksimal 10 epoch, patience 4. Checkpoint dipilih berdasarkan accuracy holdout. Perubahan arsitektur dan jadwal latihan tidak diisolasi; ini bukan ablation study.

Reproduksi: `python -m scripts.train --config configs/resnet_diversity.json --device cuda`. Gunakan environment CUDA yang dijelaskan pada README."""),
        ("code", """display(json.loads((ROOT / 'configs/resnet_diversity.json').read_text()))
display(json.loads((RUN / 'run.json').read_text()))
display(pd.read_csv(RUN / 'history.csv'))
display(Markdown((RUN / 'interpretation.md').read_text(encoding='utf-8')))"""),
        ("code", """display(Image(filename=str(RUN / 'learning_curves.png')))
display(Image(filename=str(RUN / 'confusion_matrix.png')))
display(pd.read_csv(RUN / 'confident_errors.csv').head(10))"""),
        ("markdown", """## Kandidat ensemble yang ditetapkan sebelum training
Urutan komponen adalah EfficientNet 224, EfficientNet 320, dan ResNet 224. Kandidat: ensemble lama [0,5; 0,5; 0], ResNet tunggal [0; 0; 1], dan gabungan [0,25; 0,25; 0,5]. Pilih accuracy tertinggi; jika seri, pilih kandidat yang lebih awal. Tidak ada grid search bobot atau bobot per kelas.

Cache probabilitas diselaraskan berdasarkan ID. Pipeline memeriksa class mapping, fingerprint metadata, split, hash checkpoint test, serta kesesuaian prediksi validasi dengan laporan. Arsip validasi lama belum menyimpan hash checkpoint; arsip baru menyimpannya. Keterbatasan provenance arsip lama dicatat pada selection.json."""),
        ("code", """selection = json.loads((SELECTED / 'selection.json').read_text())
display(json.loads((ROOT / 'configs/diversity_selection.json').read_text()))
display(pd.read_csv(SELECTED / 'candidates.csv'))
display(selection)
display(Markdown((SELECTED / 'interpretation.md').read_text(encoding='utf-8')))
previous = pd.read_csv(ROOT / 'reports/highres_selection/classification_report.csv', index_col=0).iloc[:16]
current = pd.read_csv(SELECTED / 'classification_report.csv', index_col=0).loc[previous.index]
comparison = pd.DataFrame({'previous_f1': previous['f1-score'], 'selected_f1': current['f1-score']})
comparison['delta_f1'] = comparison.selected_f1 - comparison.previous_f1
display(comparison.sort_values('delta_f1'))"""),
        ("markdown", """## Submission dan batas interpretasi
Holdout telah digunakan berulang untuk pemilihan epoch dan model. Peningkatan lokal belum menjadi estimasi generalisasi independen dan belum menjamin skor leaderboard meningkat. Tidak ada label test atau lookup duplikat test yang digunakan. Skor 0,86491 melekat pada file ensemble sebelumnya, bukan otomatis pada kandidat baru."""),
        ("code", """from scripts.predict import validate_submission
from scripts.preprocess import read_metadata, data_directory
_, test, sample = read_metadata(data_directory())
path = ROOT / selection['submission']
submission = pd.read_csv(path)
validate_submission(submission, sample)
assert len(submission) == len(test)
assert hashlib.sha256(path.read_bytes()).hexdigest() == selection['submission_sha256']
for row in json.loads((ROOT / 'reports/leaderboard_history.json').read_text()):
    assert hashlib.sha256((ROOT / row['submission']).read_bytes()).hexdigest() == row['submission_sha256']
print(f'Validated {len(submission)} rows; variant={selection["selected"]["variant"]}')
display(submission.head())"""),
    ])


def build_resnet_highres_notebook():
    """Create the higher-resolution ResNet continuation and four-candidate analysis."""
    write_notebook("07_resnet_highres.ipynb", [
        ("markdown", """# 07 — ResNet-50 320×320 dan pembaruan ensemble
Eksperimen sebelumnya menghasilkan accuracy validasi **87,6310%** dan macro-F1 **0,875485** melalui gabungan EfficientNet 224/320 dan ResNet 224. Skor leaderboard submission tersebut belum dilaporkan ketika tahap ini dimulai. Skor leaderboard terakhir yang diketahui, **0,86491**, berasal dari ensemble dua EfficientNet.

Hipotesis: resolusi ResNet yang lebih tinggi mempertahankan lebih banyak rincian halaman dan dapat membantu kelas scientific_report, presentation, serta form. Ini hipotesis yang diuji pada validasi, bukan jaminan peningkatan."""),
        ("code", """from pathlib import Path
import sys, json, hashlib
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'reports/resnet_highres'
SELECTED = ROOT / 'reports/resnet_highres_selection'
display(pd.read_json(ROOT / 'reports/leaderboard_history.json'))
display(pd.read_csv(ROOT / 'reports/diversity_selection/classification_report.csv', index_col=0).iloc[:16].sort_values('f1-score').head())"""),
        ("markdown", """## Melanjutkan checkpoint tanpa mengubah split
Checkpoint sumber berasal dari epoch terbaik ResNet 224 pada training split yang sama. Pipeline memeriksa fingerprint metadata, arsitektur, class mapping, dan ID validasi sebelum memuat bobot. Cache 320 berasal dari TIFF asli dengan letterbox dan padding putih; bukan upscaling cache 224.

Optimizer AdamW dimulai ulang dengan LR 0,00003, cosine schedule 6 epoch, batch 24, weight decay 0,01, label smoothing 0,05, dan patience 3. Augmentasi tetap sama: kontras ringan dan affine kecil; tanpa flip. Resolusi serta jadwal fine-tuning berubah bersama sehingga kontribusinya belum dipisahkan.

Reproduksi: `python -m scripts.train --config configs/resnet_highres.json --device cuda`. Semua layer dilatih menggunakan PyTorch; input hanya piksel."""),
        ("code", """display(json.loads((ROOT / 'configs/resnet_highres.json').read_text()))
display(json.loads((RUN / 'run.json').read_text()))
display(pd.read_csv(RUN / 'history.csv'))
display(Markdown((RUN / 'interpretation.md').read_text(encoding='utf-8')))"""),
        ("code", """display(Image(filename=str(RUN / 'learning_curves.png')))
display(Image(filename=str(RUN / 'confusion_matrix.png')))
display(pd.read_csv(RUN / 'class_comparison.csv').sort_values('delta_f1'))"""),
        ("markdown", """## Empat kandidat ditetapkan sebelum training
Urutan komponen: EfficientNet 224, EfficientNet 320, ResNet 224, ResNet 320. Kandidat pertama adalah ensemble sebelumnya dengan bobot [0,25; 0,25; 0,50; 0]. Pembandingnya adalah ResNet 320 tunggal, penggantian komponen ResNet 224 dengan ResNet 320, dan rata-rata empat model. Semua bobot tersedia pada konfigurasi di bawah.

Pilih accuracy validasi tertinggi; seri memilih kandidat paling awal. Tidak ada tuning bobot per kelas, label test, atau lookup gambar duplikat. Pemilihan checkpoint dan empat kandidat ini menambah penggunaan holdout untuk tuning, sehingga skor belum menjadi estimasi independen."""),
        ("code", """selection = json.loads((SELECTED / 'selection.json').read_text())
display(json.loads((ROOT / 'configs/resnet_highres_selection.json').read_text()))
display(pd.read_csv(SELECTED / 'candidates.csv'))
display(Markdown((SELECTED / 'interpretation.md').read_text(encoding='utf-8')))
display(pd.read_csv(SELECTED / 'class_comparison.csv').sort_values('delta_f1'))
display(pd.read_csv(SELECTED / 'confident_errors.csv').head(10))"""),
        ("markdown", """## Validasi submission
File terpilih tetap mengikuti urutan ID sample submission. Hash checkpoint, arsip validasi, konfigurasi kandidat, dan submission dicatat pada selection.json. Arsip validasi EfficientNet lama belum menyimpan hash checkpoint; konsistensinya diperiksa melalui prediksi tercatat. Arsip kedua ResNet memiliki hash checkpoint.

Skor holdout dan leaderboard dibedakan. Submission terdahulu dipertahankan agar perubahan hasil bisa ditelusuri."""),
        ("code", """from scripts.predict import validate_submission
from scripts.preprocess import read_metadata, data_directory
_, test, sample = read_metadata(data_directory())
path = ROOT / selection['submission']
submission = pd.read_csv(path)
validate_submission(submission, sample)
assert len(submission) == len(test)
assert hashlib.sha256(path.read_bytes()).hexdigest() == selection['submission_sha256']
previous = json.loads((ROOT / 'reports/diversity_selection/selection.json').read_text())
assert hashlib.sha256((ROOT / previous['submission']).read_bytes()).hexdigest() == previous['submission_sha256']
for row in json.loads((ROOT / 'reports/leaderboard_history.json').read_text()):
    assert hashlib.sha256((ROOT / row['submission']).read_bytes()).hexdigest() == row['submission_sha256']
print(f'Validated {len(submission)} rows; variant={selection["selected"]["variant"]}')
display(submission.head())"""),
    ])


def build_convnext_notebook():
    """Document the new ImageNet backbone and fixed five-component ensemble candidates."""
    write_notebook("08_convnext.ipynb", [
        ("markdown", """# 08 — ConvNeXt-Tiny dan evaluasi ensemble
Ensemble empat model sebelumnya mencapai accuracy holdout **88,2252%** dan macro-F1 **0,881401**. Skor leaderboard submission itu belum dilaporkan saat eksperimen ini dimulai. Kelas terlemah tetap scientific_report, presentation, dan form.

Hipotesis: ConvNeXt dapat menghasilkan kesalahan yang berbeda dari EfficientNet dan ResNet sehingga membantu ensemble. Inisialisasi menggunakan [ConvNeXt-Tiny ImageNet-1K V1 resmi Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), tanpa checkpoint yang dilatih pada dataset kompetisi ini. Semua label supervised berasal dari training split yang sama."""),
        ("code", """from pathlib import Path
import sys, json, hashlib
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'reports/convnext_diversity'
SELECTED = ROOT / 'reports/convnext_selection'
display(pd.read_json(ROOT / 'reports/leaderboard_history.json'))
display(pd.read_csv(ROOT / 'reports/resnet_highres_selection/classification_report.csv', index_col=0).iloc[:16].sort_values('f1-score').head())"""),
        ("markdown", """## Preprocessing dan training
Cache letterbox 224×224 mempertahankan rasio halaman dan padding putih. Grayscale direplikasi ke tiga kanal dan dinormalisasi ImageNet. Training memakai kontras dan affine ringan seperti eksperimen sebelumnya; validasi serta inference deterministik. Tidak ada OCR, fitur ID/filename, flip, atau lookup label test.

Head diganti menjadi 16 kelas; semua layer di-fine-tune dengan AdamW LR 0,0001, weight decay 0,05, cosine schedule 10 epoch, label smoothing 0,05, batch 24, dan patience 4. Stochastic depth menggunakan default ConvNeXt-Tiny torchvision (0,1). Konfigurasi optimizer berbeda dari backbone sebelumnya, sehingga ini bukan ablation arsitektur murni.

Reproduksi: `python -m scripts.train --config configs/convnext_diversity.json --device cuda`. Bobot pertama diunduh dari PyTorch, diverifikasi hash-nya, lalu disimpan pada checkpoint/pretrained yang diabaikan Git."""),
        ("code", """display(json.loads((ROOT / 'configs/convnext_diversity.json').read_text()))
display(json.loads((RUN / 'run.json').read_text()))
display(pd.read_csv(RUN / 'history.csv'))
display(Markdown((RUN / 'interpretation.md').read_text(encoding='utf-8')))"""),
        ("code", """display(Image(filename=str(RUN / 'learning_curves.png')))
display(Image(filename=str(RUN / 'confusion_matrix.png')))
display(pd.read_csv(RUN / 'confident_errors.csv').head(10))"""),
        ("markdown", """## Kandidat berbobot tetap
Urutan komponen: EfficientNet 224, EfficientNet 320, ResNet 224, ResNet 320, ConvNeXt 224. Empat kandidat ditetapkan sebelum training: ensemble lama, ConvNeXt tunggal, kombinasi dengan total bobot 1/3 untuk setiap keluarga arsitektur, serta kombinasi 50:50 ensemble lama dengan ConvNeXt.

Pilih accuracy holdout tertinggi; seri memilih kandidat paling awal. Tidak ada grid search bobot atau bobot per kelas. Pipeline memeriksa split lengkap, ID, label mapping, fingerprint metadata, dan identitas checkpoint pada cache probabilitas. Arsip validasi EfficientNet lama belum menyimpan hash checkpoint; keterbatasan ini dicatat pada laporan komponen."""),
        ("code", """selection = json.loads((SELECTED / 'selection.json').read_text())
display(json.loads((ROOT / 'configs/convnext_selection.json').read_text()))
display(pd.read_csv(SELECTED / 'candidates.csv'))
display(Markdown((SELECTED / 'interpretation.md').read_text(encoding='utf-8')))
display(pd.read_csv(SELECTED / 'class_comparison.csv').sort_values('delta_f1'))
display(pd.read_csv(SELECTED / 'confident_errors.csv').head(10))"""),
        ("markdown", """## Submission dan batas kesimpulan
Holdout sudah dipakai berulang untuk memilih checkpoint dan kandidat. Skor lokal ini belum menjadi estimasi independen setelah tuning; hasil leaderboard perlu dicatat secara terpisah. Jika kandidat baru tidak membaik, selector mempertahankan ensemble sebelumnya. Tidak ada label test yang digunakan untuk pemilihan.

Seluruh submission terdahulu dipertahankan. Bobot pretrained dan checkpoint dapat melebihi 100 MB sehingga wajib tetap berada pada direktori yang diabaikan Git, bukan diunggah sebagai source."""),
        ("code", """from scripts.predict import validate_submission
from scripts.preprocess import read_metadata, data_directory
_, test, sample = read_metadata(data_directory())
path = ROOT / selection['submission']
submission = pd.read_csv(path)
validate_submission(submission, sample)
assert len(submission) == len(test)
assert hashlib.sha256(path.read_bytes()).hexdigest() == selection['submission_sha256']
for name in ('highres_selection', 'diversity_selection', 'resnet_highres_selection'):
    previous = json.loads((ROOT / 'reports' / name / 'selection.json').read_text())
    assert hashlib.sha256((ROOT / previous['submission']).read_bytes()).hexdigest() == previous['submission_sha256']
print(f'Validated {len(submission)} rows; variant={selection["selected"]["variant"]}')
display(submission.head())"""),
    ])


def main():
    """Generate EDA, modeling, and submission notebooks with interpretation prompts."""
    setup = '''from pathlib import Path
import sys
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent
assert (ROOT / 'scripts').is_dir(), 'Buka notebook dari root proyek atau notebooks/'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.preprocess import data_directory, read_metadata
DATA = data_directory()
train, test, sample = read_metadata(DATA)'''
    write_notebook("01_eda.ipynb", [
        ("markdown", """# 01 — Problem understanding dan EDA
Klasifikasi **visual** halaman dokumen ke 16 kelas menggunakan PyTorch. Input model hanya piksel; nama file, ID, dan label contoh submission bukan fitur.

Alur mengikuti proyek referensi: EDA → modeling → submission, dengan implementasi bersama di `scripts/`. Bobot penilaian repositori adalah 70%, public leaderboard 10%, dan private leaderboard 20%. Nama metrik leaderboard belum tersedia; laporkan accuracy dan macro-F1 secara terpisah."""),
        ("code", setup),
        ("markdown", "## Struktur dan kualitas metadata\nPeriksa jumlah baris, nilai kosong, keunikan ID, dan pemetaan label. Loader mendukung ekstraksi normal maupun folder `train/train` dan `test/test`."),
        ("code", "display(train.head())\ndisplay(pd.DataFrame({'rows': [len(train), len(test)], 'unique_ids': [train.id.nunique(), test.id.nunique()]}, index=['train', 'test']))\ndisplay(train.isna().sum().to_frame('missing'))"),
        ("markdown", "## Audit seluruh gambar\nDecode semua TIFF; hitung dimensi, jumlah halaman, dan hash piksel. Gambar training yang tidak terbaca dicatat dan dikeluarkan dari kedua split; gambar test yang tidak terbaca menghentikan pipeline. Cache menggunakan grayscale dan padding putih, tanpa memotong teks."),
        ("code", "from scripts.preprocess import prepare_data, verified_cache\nif not (ROOT / 'data/cache_128/manifest.json').exists():\n    prepare_data(DATA, image_size=128)\nverified_cache(DATA, 128)\naudit = json.loads((ROOT / 'reports/data_audit.json').read_text())\ndisplay({key: audit[key] for key in ['train_rows', 'test_rows', 'usable_training_rows', 'unreadable_images', 'excluded_training_images', 'train_duplicate_pixel_rows', 'conflicting_duplicate_groups', 'test_images_matching_train_pixels', 'multipage_images']})"),
        ("markdown", "## Distribusi target\nBandingkan kelas terbanyak dan tersedikit sebelum memutuskan perlunya class weighting. Bobot kelas tidak diterapkan otomatis pada dataset yang hampir seimbang."),
        ("code", "counts = train.groupby(['label', 'class_name']).size().rename('count')\ndisplay(counts.to_frame())\nprint(f'Rasio kelas terbesar/terkecil: {counts.max()/counts.min():.3f}')\ndisplay(Image(filename=str(ROOT / 'reports/figures/class_distribution.png')))"),
        ("markdown", "## Sampel visual dan rasio aspek\nPerhatikan layout kolom, tabel, kop surat, serta dokumen landscape. Pada resolusi 128, teks kecil kehilangan detail; baseline terutama menguji informasi layout."),
        ("code", "display(Image(filename=str(ROOT / 'reports/figures/sample_documents.png')))\ndisplay(Image(filename=str(ROOT / 'reports/figures/aspect_ratios.png')))"),
        ("markdown", "## Interpretasi dan keputusan preprocessing\nKelas relatif seimbang, sehingga accuracy berguna sebagai ringkasan dan macro-F1 memeriksa pemerataan performa kelas. Padding mempertahankan rasio aspek dan isi halaman. Hindari flip dan crop agresif karena dapat menghilangkan header atau mengubah arah baca. Kontras ringan hanya diterapkan pada training. TIFF multipage menggunakan halaman pertama.\n\nHash piksel yang identik menjadi satu grup split. Ini mencegah kebocoran duplikat persis, tetapi tidak menjamin halaman dari dokumen sumber yang sama selalu bersama: ID dokumen sumber tidak tersedia. Nama file tidak diasumsikan sebagai ID grup dokumen. Test digunakan untuk audit teknis dan prediksi, tidak untuk pemilihan model.")
    ])
    write_notebook("02_modeling.ipynb", [
        ("markdown", """# 02 — Preprocessing, modeling, dan evaluasi
Baseline adalah CNN PyTorch dari awal, diikuti opsi fine-tuning EfficientNet-B0. Gunakan split yang sama untuk perbandingan. Semua metrik yang ditampilkan berasal dari artefak training, bukan angka perkiraan."""),
        ("code", setup),
        ("markdown", "## Konfigurasi dan perangkat\nCNN baseline menggunakan 128×128 grayscale, empat blok konvolusi, serta grid 4×4 sebelum classifier. EfficientNet menggunakan 320×320, tiga kanal grayscale, normalisasi ImageNet, dan head 16 kelas. Unduhan bobot pretrained memerlukan internet."),
        ("code", "import torch\nfrom scripts.model import build_model\nCONFIG = ROOT / 'configs/baseline.json'\nconfig = json.loads(CONFIG.read_text())\ndisplay(config)\nprint('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\nmodel = build_model(config['model'], pretrained=False)\nprint(model)\nprint('Parameters:', sum(p.numel() for p in model.parameters()))"),
        ("markdown", "## Protokol validasi\nStratifiedGroupKFold membagi grup hash gambar ke lima fold; fold pertama menjadi holdout sekitar 20%. Gambar rusak dikecualikan. Seluruh training dan validasi memiliki 16 kelas. Holdout digunakan untuk checkpoint selection sehingga bukan estimasi final yang independen dari tuning. Seed 42 disimpan bersama konfigurasi dan split."),
        ("code", "from scripts.preprocess import verified_cache, make_split\ncache = verified_cache(DATA, config['image_size'])\nimage_audit = pd.read_csv(cache / 'train_audit.csv')\nfit_idx, val_idx = make_split(train, image_audit, config['seed'], config['validation_fraction'])\nassert not set(image_audit.iloc[fit_idx].sha256) & set(image_audit.iloc[val_idx].sha256)\ndisplay(pd.concat([train.iloc[fit_idx].label.value_counts().sort_index().rename('train'), train.iloc[val_idx].label.value_counts().sort_index().rename('validation')], axis=1))\nprint('Train:', len(fit_idx), 'Validation:', len(val_idx))"),
        ("markdown", "## Training\nAdamW, cosine learning rate, label smoothing 0,05, gradient clipping, dan early stopping. CUDA memakai mixed precision jika tersedia. Checkpoint dipilih berdasarkan `selection_metric` (default accuracy, belum dikonfirmasi sebagai metrik leaderboard). Run yang sudah ada dibaca tanpa ditimpa. Untuk eksperimen baru, ubah konfigurasi dan nama run melalui CLI."),
        ("code", "from scripts.train import run_training\nREPORT = ROOT / 'reports' / config['run_name']\nif not (REPORT / 'metrics.json').exists():\n    run_training(CONFIG, DATA)\nmetrics = json.loads((REPORT / 'metrics.json').read_text())\ndisplay({key: value for key, value in metrics.items() if key != 'classification_report'})\ndisplay(pd.read_csv(REPORT / 'history.csv'))"),
        ("markdown", "## Learning curves dan confusion matrix\nTraining loss memakai label smoothing, validation loss tidak; nilai absolut keduanya tidak sepenuhnya sebanding. Periksa tren learning curve dan perbedaan performa antar kelas."),
        ("code", "display(Image(filename=str(REPORT / 'learning_curves.png')))\ndisplay(Image(filename=str(REPORT / 'confusion_matrix.png')))\nclass_report = pd.DataFrame(metrics['classification_report']).T\ndisplay(class_report)"),
        ("markdown", "## Analisis kesalahan\nLihat kelas dengan F1 terendah dan prediksi salah dengan confidence tinggi. Kemiripan layout dapat menjadi hipotesis kesalahan; periksa halaman sumber sebelum menyimpulkan penyebab. Confidence softmax bukan probabilitas yang sudah dikalibrasi."),
        ("code", "from scripts.preprocess import CLASSES, resolve_image\nfrom PIL import Image as PILImage\ndisplay(class_report.loc[CLASSES].sort_values('f1-score').head(5))\nerrors = pd.read_csv(REPORT / 'confident_errors.csv')\ndisplay(errors.head(10))\nfor row in errors.head(3).itertuples():\n    print(f'{row.id}: aktual={row.class_name}, prediksi={row.predicted_class}, confidence={row.confidence:.3f}')\n    with PILImage.open(resolve_image(DATA, row.filename)) as page:\n        preview = page.convert('RGB')\n        preview.thumbnail((450, 550))\n        display(preview)"),
        ("markdown", "## Eksperimen berikutnya\nJalankan konfigurasi EfficientNet pada GPU setelah baseline. Pertahankan split, bandingkan accuracy dan macro-F1, lalu putuskan berdasarkan metrik resmi kompetisi setelah tersedia. Resolusi 320 mempertahankan lebih banyak detail, tetapi keunggulannya harus dibuktikan lewat validasi. Hindari memakai bobot yang telah dilatih pada dataset dokumen kompetisi tanpa kepastian aturan dan pemisahan data. Public/private score hanya dicatat sesudah hasil resmi tersedia.")
    ])
    write_notebook("03_submission.ipynb", [
        ("markdown", "# 03 — Inference dan submission\nGunakan checkpoint terbaik dan preprocessing yang sama dengan training. Prediksi dipetakan berdasarkan ID, kemudian diurutkan mengikuti sample submission. Nilai nol di sample submission hanyalah contoh format."),
        ("code", setup),
        ("markdown", "## Prediksi test\nCheckpoint baseline menggunakan training split saja; model belum di-refit pada seluruh label. Jika submission sudah tersedia, baca hasilnya. Ubah `REGENERATE` menjadi True untuk inferensi ulang."),
        ("code", "from scripts.predict import generate_submission, validate_submission\nCHECKPOINT = ROOT / 'checkpoint/baseline_cnn/best.pt'\nOUTPUT = ROOT / 'submissions/submission_baseline.csv'\nREGENERATE = False\nif REGENERATE or not OUTPUT.exists():\n    assert CHECKPOINT.exists(), 'Jalankan notebook modeling atau scripts.train terlebih dahulu'\n    submission = generate_submission(CHECKPOINT, OUTPUT, DATA)\nelse:\n    submission = pd.read_csv(OUTPUT)\ndisplay(submission.head())"),
        ("markdown", "## Validasi kontrak submission\nHarus tepat dua kolom: `id,label`; satu baris per ID, tanpa nilai kosong, dengan label integer 0–15. Urutan harus sesuai template."),
        ("code", "validate_submission(submission, sample)\nassert len(submission) == len(test)\nprint(f'Valid: {len(submission):,} baris, kolom {submission.columns.tolist()}')\ndisplay(submission.label.value_counts().sort_index().rename('predicted_count').to_frame())\nprint(OUTPUT)"),
        ("markdown", "## Provenance dan pencatatan leaderboard\nHash SHA-256 mengaitkan submission dengan checkpoint. Skor validasi lokal bukan skor leaderboard. Upload dilakukan manual melalui [halaman kompetisi](https://www.kaggle.com/competitions/document-image-classification-sd234504/data) menggunakan akun yang memiliki akses. Metrik resmi belum dapat diverifikasi karena halaman tidak tersedia pada sesi pemeriksaan."),
        ("code", "import hashlib\nprovenance_path = ROOT / 'reports/submission_baseline_provenance.json'\nprovenance = json.loads(provenance_path.read_text())\nassert hashlib.sha256(OUTPUT.read_bytes()).hexdigest() == provenance['submission_sha256']\ndisplay(provenance)"),
        ("markdown", "## Checklist pelaporan\nCatat nama file, run, epoch terbaik, skor validasi, dan skor resmi setelah submission. Dataset, cache, checkpoint, dan submission lokal diabaikan oleh `.gitignore`. Commit source, notebook yang sudah dieksekusi, konfigurasi, dan laporan kecil. Jalankan `python -m scripts.validate_repository` sebelum mengunggah repository.")
    ])


if __name__ == "__main__":
    main()
