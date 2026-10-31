"""Extract local English OCR from pixel-only TIFF headers into resumable ID-keyed chunks."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import pandas as pd

from scripts.preprocess import ROOT, data_directory, metadata_digest, read_metadata, resolve_image


CONFIG = ROOT / "configs/ocr_header.json"
HELPER = ROOT / "scripts/read_header_ocr.ps1"


def cache_folder(config, directory):
    """Bind the cache to the input metadata, crop recipe, OCR helper and Windows version."""
    identity = [metadata_digest(directory), config["language"], config["header_fraction"], config["scale"],
                platform.version(), hashlib.sha256(HELPER.read_bytes()).hexdigest()]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return ROOT / "data/ocr_cache" / key


def read_chunk(path, identifiers):
    """Require one complete UTF-8 line for every expected ID, including empty text."""
    with path.open(encoding="utf-8") as stream:
        rows = [line.rstrip("\r\n").split("\t", 1) for line in stream]
    if any(len(row) != 2 for row in rows) or [row[0] for row in rows] != identifiers:
        raise ValueError(f"Incomplete or misaligned OCR chunk {path.name}")
    return [text for _, text in rows]


def extract_chunk(start, frame, directory, folder):
    """Read one bounded batch through Windows OCR and atomically publish verified text."""
    name = f"chunk_{start:06d}"
    output = folder / (name + ".tsv")
    ids = frame.id.tolist()
    if output.exists():
        read_chunk(output, ids)
        return len(frame)
    manifest = folder / (name + ".paths")
    temporary = folder / (name + ".partial")
    with manifest.open("w", encoding="utf-8") as stream:
        for row in frame.itertuples():
            stream.write(f"{row.id}\t{resolve_image(directory, row.filename)}\n")
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "RemoteSigned", "-File", str(HELPER),
                        "-Manifest", str(manifest), "-OutputFile", str(temporary), "-Config", str(CONFIG)],
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        read_chunk(temporary, ids)
        temporary.replace(output)
    finally:
        manifest.unlink(missing_ok=True)
    return len(frame)


def input_rows(directory):
    """Select only readable training images and all test IDs in stable metadata order."""
    train, test, _ = read_metadata(directory)
    split = pd.read_csv(ROOT / "reports/convnext_diversity/split.csv").set_index("id").loc[train.id]
    assert split.index.is_unique and split.label.tolist() == train.label.tolist()
    usable = train.loc[split.split.isin(["train", "validation"]).to_numpy(), ["id", "filename"]]
    return pd.concat([usable, test[["id", "filename"]]], ignore_index=True)


def run():
    """Extract the dataset in parallel and write an ID-verified cache completion marker."""
    if os.name != "nt":
        raise RuntimeError("Header extraction requires Windows with an English OCR language pack")
    config = json.loads(CONFIG.read_text())
    directory = data_directory()
    rows = input_rows(directory)
    folder = cache_folder(config, directory)
    folder.mkdir(parents=True, exist_ok=True)
    count, started = 0, time.perf_counter()
    size = config["chunk_size"]
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        jobs = [pool.submit(extract_chunk, i, rows.iloc[i:i + size], directory, folder)
                for i in range(0, len(rows), size)]
        for job in as_completed(jobs):
            count += job.result()
            if count // size % 4 == 0 or count == len(rows):
                print(f"OCR {count}/{len(rows)} in {time.perf_counter()-started:.0f}s", flush=True)
    (folder / "complete.json").write_text(json.dumps({"rows": count, "chunk_size": size,
        "metadata_digest": metadata_digest(directory), "helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        "language": config["language"], "os": platform.version()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
