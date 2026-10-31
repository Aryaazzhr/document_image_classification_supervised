"""Validate metadata, audit TIFF images, and cache complete document layouts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "advertisement", "budget", "email", "file_folder", "form", "handwritten",
    "invoice", "letter", "memo", "news_article", "presentation", "questionnaire",
    "resume", "scientific_publication", "scientific_report", "specification",
]


def data_directory(value=None):
    """Find the metadata directory, preferring data/ over the original root."""
    if value is not None:
        return Path(value).resolve()
    return ROOT / "data" if (ROOT / "data/train.csv").exists() else ROOT


def read_metadata(data_dir):
    """Read and validate schemas, identifiers, class mapping, and submission IDs."""
    frames = [pd.read_csv(Path(data_dir) / name) for name in
              ("train.csv", "test.csv", "sample_submission.csv")]
    train, test, sample = frames
    expected = [["id", "filename", "class_name", "label"], ["id", "filename"], ["id", "label"]]
    for frame, columns in zip(frames, expected):
        if frame.columns.tolist() != columns or frame.isna().any().any():
            raise ValueError(f"Invalid columns or missing values; expected {columns}")
        if not frame.id.is_unique:
            raise ValueError("Duplicate metadata IDs")
    if not pd.api.types.is_integer_dtype(train.label) or not train.label.between(0, 15).all():
        raise ValueError("Training labels must be integers 0..15")
    if not train.class_name.eq(train.label.map(dict(enumerate(CLASSES)))).all():
        raise ValueError("Training class_name does not match the competition label mapping")
    if set(train.label) != set(range(16)):
        raise ValueError("Expected all 16 classes in the training metadata")
    if not train.filename.is_unique or not test.filename.is_unique:
        raise ValueError("Duplicate filenames in metadata")
    if set(test.id) != set(sample.id):
        raise ValueError("Test and sample submission IDs differ")
    return train, test, sample


def resolve_image(data_dir, filename):
    """Resolve both train/class/file.tif and nested train/train/class/file.tif."""
    base = Path(data_dir).resolve()
    relative = Path(str(filename).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"train", "test"}:
        raise ValueError(f"Unsafe or unexpected image path: {filename}")
    for path in (base / relative, base / relative.parts[0] / relative):
        if path.is_file() and path.resolve().is_relative_to(base):
            return path
    raise FileNotFoundError(f"Image missing: {filename} under {base}")


def metadata_digest(data_dir):
    """Fingerprint both model input CSVs to prevent stale cache/row alignment."""
    digest = hashlib.sha256()
    for name in ("train.csv", "test.csv"):
        digest.update((Path(data_dir) / name).read_bytes())
    return digest.hexdigest()


def load_page(path, size):
    """Decode the first TIFF page and pad to square without cropping text."""
    with Image.open(path) as image:
        image.seek(0)
        page = ImageOps.exif_transpose(image).convert("L")
        original_size = page.size
        digest = hashlib.sha256(str(original_size).encode() + page.tobytes()).hexdigest()
        resized = ImageOps.pad(page, (size, size), method=Image.Resampling.BILINEAR, color=255)
        return np.asarray(resized), original_size, image.n_frames, digest


def inspect_job(job):
    """Decode one cache job in a worker thread and return metadata or an error."""
    index, path, size = job
    try:
        array, dimensions, pages, digest = load_page(path, size)
        return index, array, dimensions, pages, digest, None
    except Exception as error:
        return index, None, None, None, None, f"{path}: {error}"


def prepare_data(data_dir=None, image_size=128, workers=8):
    """Audit every image, detect exact duplicates, and persist reusable uint8 caches."""
    data_dir = data_directory(data_dir)
    train, test, _ = read_metadata(data_dir)
    cache_dir = ROOT / "data" / f"cache_{image_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").unlink(missing_ok=True)
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    audit_frames, all_errors = [], []
    for name, frame in (("train", train), ("test", test)):
        paths = [resolve_image(data_dir, filename) for filename in frame.filename]
        temporary = cache_dir / f"{name}.partial.npy"
        cache = np.lib.format.open_memmap(temporary, mode="w+", dtype="uint8",
                                        shape=(len(frame), image_size, image_size))
        records, errors = [], []
        jobs = ((i, path, image_size) for i, path in enumerate(paths))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, array, dimensions, pages, digest, error in pool.map(inspect_job, jobs):
                if error:
                    errors.append(error)
                    cache[i] = 255
                    records.append({"id": frame.iloc[i].id, "width": None, "height": None,
                                    "pages": 0, "sha256": f"unreadable:{frame.iloc[i].id}", "valid": False})
                    continue
                cache[i] = array
                records.append({"id": frame.iloc[i].id, "width": dimensions[0],
                                "height": dimensions[1], "pages": pages, "sha256": digest, "valid": True})
                if (i + 1) % 5000 == 0:
                    print(f"{name}: {i + 1}/{len(frame)} decoded", flush=True)
        cache.flush()
        del cache
        all_errors.extend(errors)
        (report_dir / "image_errors.json").write_text(json.dumps(all_errors, indent=2), encoding="utf-8")
        if errors and name == "test":
            raise ValueError(f"{len(errors)} unreadable images; see reports/image_errors.json")
        temporary.replace(cache_dir / f"{name}.npy")
        audit = pd.DataFrame(records)
        audit.to_csv(cache_dir / f"{name}_audit.csv", index=False)
        audit_frames.append(audit)
    train_audit, test_audit = audit_frames
    export_duplicate_report(train, train_audit)
    conflicts = train.assign(sha256=train_audit.sha256).groupby("sha256").label.nunique().gt(1).sum()
    summary = {
        "train_rows": len(train), "test_rows": len(test), "classes": CLASSES,
        "class_counts": train.class_name.value_counts().reindex(CLASSES).to_dict(),
        "missing_metadata_values": int(train.isna().sum().sum() + test.isna().sum().sum()),
        "unreadable_images": len(all_errors),
        "excluded_training_images": train_audit.loc[~train_audit.valid, "id"].tolist(),
        "usable_training_rows": int(train_audit.valid.sum()),
        "train_duplicate_pixel_rows": int(train_audit.sha256.duplicated().sum()),
        "conflicting_duplicate_groups": int(conflicts),
        "test_images_matching_train_pixels": int(test_audit.sha256.isin(train_audit.sha256).sum()),
        "multipage_images": int(sum(a.pages.gt(1).sum() for a in audit_frames)),
        "dimensions": {name: audit[["width", "height"]].describe().to_dict()
                       for name, audit in zip(("train", "test"), audit_frames)},
    }
    (report_dir / "data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"image_size": image_size, "metadata_digest": metadata_digest(data_dir),
                "train_rows": len(train), "test_rows": len(test), "preprocessing": "grayscale_letterbox_v1"}
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_eda_figures(train, train_audit, cache_dir)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def export_duplicate_report(train, audit):
    """List exact duplicate training pages and preserve conflicting source labels."""
    combined = train.assign(sha256=audit.sha256.to_numpy())
    duplicates = combined.loc[combined.sha256.duplicated(keep=False)].copy()
    duplicates["conflicting_labels"] = duplicates.groupby("sha256").label.transform("nunique").gt(1)
    duplicates.sort_values(["sha256", "id"]).to_csv(ROOT / "reports/duplicate_training_pages.csv", index=False)
    return duplicates


def verified_cache(data_dir, image_size):
    """Require a completed cache whose metadata fingerprint and shape match inputs."""
    cache_dir = ROOT / "data" / f"cache_{image_size}"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run python -m scripts.preprocess --image-size {image_size} first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["metadata_digest"] != metadata_digest(data_dir):
        raise ValueError("Metadata changed after caching. Re-run preprocessing.")
    for split in ("train", "test"):
        array = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
        if array.shape != (manifest[f"{split}_rows"], image_size, image_size):
            raise ValueError("Cache shape does not match manifest")
    return cache_dir


def make_split(train, audit, seed=42, validation_fraction=0.2):
    """Use a stratified group holdout so identical pages cannot cross the split."""
    folds = round(1 / validation_fraction)
    if folds < 2 or not np.isclose(1 / folds, validation_fraction):
        raise ValueError("validation_fraction must be 1/k for integer k >= 2")
    if train.id.tolist() != audit.id.tolist():
        raise ValueError("Audit rows do not match training IDs")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    eligible = np.flatnonzero(audit.valid.to_numpy())
    fit, holdout = next(splitter.split(train.iloc[eligible], train.iloc[eligible].label,
                                      groups=audit.iloc[eligible].sha256))
    training, validation = eligible[fit], eligible[holdout]
    if set(audit.iloc[training].sha256) & set(audit.iloc[validation].sha256):
        raise RuntimeError("Duplicate-image leakage detected")
    if any(set(train.iloc[index].label) != set(range(16)) for index in (training, validation)):
        raise ValueError("Both splits must contain all 16 classes")
    return training, validation


def make_eda_figures(train, audit, cache_dir):
    """Export class balance, image dimensions, and one document per category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = ROOT / "reports/figures"
    target.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    train.class_name.value_counts().reindex(CLASSES).plot.barh(ax=ax, color="#286b8f")
    ax.set(xlabel="Jumlah halaman", title="Distribusi 16 kelas training")
    fig.tight_layout()
    fig.savefig(target / "class_distribution.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(12, 14))
    images = np.load(cache_dir / "train.npy", mmap_mode="r")
    for label, ax in enumerate(axes.flat):
        index = train.index[train.label.eq(label) & audit.valid][0]
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{label}: {CLASSES[label]}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(target / "sample_documents.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(audit.width / audit.height, bins=40, color="#286b8f")
    ax.set(xlabel="Lebar / tinggi", ylabel="Jumlah halaman", title="Rasio aspek halaman training")
    fig.tight_layout()
    fig.savefig(target / "aspect_ratios.png", dpi=140)
    plt.close(fig)


def main():
    """Parse CLI options for the full image audit and preprocessing workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.image_size < 32 or args.workers < 1:
        parser.error("image-size >= 32 and workers >= 1 are required")
    prepare_data(args.data_dir, args.image_size, args.workers)


if __name__ == "__main__":
    main()
