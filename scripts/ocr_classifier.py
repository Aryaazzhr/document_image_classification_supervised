"""Learn document classes from OCR of scanned headers and evaluate a fixed visual/text blend."""

import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
import torch
from torch import nn
from torch.nn import functional as F

from scripts.extract_ocr import CONFIG, cache_folder, input_rows, read_chunk
from scripts.predict import validate_submission
from scripts.preprocess import CLASSES, ROOT, data_directory, metadata_digest, read_metadata
from scripts.select_ensemble import load_component


def clean_header(text):
    """Normalize OCR whitespace and mask digit strings such as page serial numbers."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", text.casefold())).strip()


def word_count(text):
    """Count recognized alphabetic words, ignoring punctuation and serial numbers."""
    return len(re.findall(r"\b[a-z]{2,}\b", text))


def read_headers(config, directory):
    """Load completed OCR chunks and require a unique text record for each usable page."""
    rows = input_rows(directory)
    cache = cache_folder(config, directory)
    marker = json.loads((cache / "complete.json").read_text())
    if marker["rows"] != len(rows) or marker["metadata_digest"] != metadata_digest(directory):
        raise ValueError("OCR cache completion marker differs from the dataset")
    headers = []
    size = marker["chunk_size"]
    for start in range(0, len(rows), size):
        identifiers = rows.id.iloc[start:start + size].tolist()
        headers.extend(read_chunk(cache / f"chunk_{start:06d}.tsv", identifiers))
    if not rows.id.is_unique:
        raise ValueError("OCR IDs must be unique across both partitions")
    return pd.Series([clean_header(text) for text in headers], index=rows.id), cache.name


class HeaderClassifier(nn.Module):
    """Pool learned word and character n-gram embeddings into 16 supervised class logits."""

    def __init__(self, width, hidden):
        """Initialize the weighted embedding bag and its nonlinear classification head."""
        super().__init__()
        self.terms = nn.EmbeddingBag(width, hidden, mode="sum", include_last_offset=True)
        nn.init.normal_(self.terms.weight, std=0.06)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(hidden, len(CLASSES)))

    def forward(self, ids, offsets, values):
        """Use the row's TF-IDF values as weights for its term embeddings."""
        return self.head(self.terms(ids, offsets, per_sample_weights=values))


def tensors(matrix, device):
    """Represent a sparse batch as term IDs, document offsets, and positive weights."""
    return (torch.from_numpy(matrix.indices.astype(np.int64)).to(device),
            torch.from_numpy(matrix.indptr.astype(np.int64)).to(device),
            torch.from_numpy(matrix.data).to(device))


def transform(vectorizers, texts):
    """Build sparse features using only the vocabularies fitted to training headers."""
    return sparse.hstack([v.transform(texts) for v in vectorizers], format="csr", dtype=np.float32)


def train_head(texts, labels, config, device, folder, signature):
    """Fit TF-IDF and a fixed-epoch PyTorch classifier exclusively on training rows."""
    path = folder / "head.pt"
    vocab = folder / "vocabulary.joblib"
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["signature"] != signature or saved["vocab_hash"] != hashlib.sha256(vocab.read_bytes()).hexdigest():
            raise ValueError("Cached OCR model or vocabulary belongs to a different training run")
        vectorizers = joblib.load(vocab)
        model = HeaderClassifier(saved["width"], config["hidden"]).to(device)
        model.load_state_dict(saved["weights"])
        return model, vectorizers
    vectorizers = [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=config["max_word_features"],
                        sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=config["max_char_features"],
                        sublinear_tf=True, dtype=np.float32)]
    print("Fitting word and character vocabularies on training headers", flush=True)
    matrix = sparse.hstack([v.fit_transform(texts) for v in vectorizers], format="csr", dtype=np.float32)
    model = HeaderClassifier(matrix.shape[1], config["hidden"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    target = torch.as_tensor(labels.to_numpy(), device=device)
    rng = np.random.default_rng(config["seed"])
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        order = rng.permutation(len(texts))
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(*tensors(matrix[batch], device))
            loss = F.cross_entropy(outputs, target[torch.as_tensor(batch, device=device)], label_smoothing=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        scheduler.step()
        history.append({"epoch": epoch + 1, "training_loss": total / len(texts)})
        print(f"Epoch {epoch+1}/{config['epochs']}, loss {total / len(texts):.4f}", flush=True)
    report = ROOT / "reports" / config["run_name"]
    pd.DataFrame(history).to_csv(report / "history.csv", index=False)
    joblib.dump(vectorizers, vocab)
    partial = folder / "head.partial.pt"
    torch.save({"weights": model.state_dict(), "signature": signature, "width": matrix.shape[1],
                "classes": CLASSES, "config": config, "vocab_hash": hashlib.sha256(vocab.read_bytes()).hexdigest()}, partial)
    partial.replace(path)
    return model, vectorizers


def probabilities(model, vectorizers, texts, device):
    """Predict probabilities in order, including documents with empty OCR."""
    matrix = transform(vectorizers, texts)
    model.eval()
    with torch.inference_mode():
        parts = [model(*tensors(matrix[start:start + 256], device)).softmax(1).cpu().numpy()
                 for start in range(0, matrix.shape[0], 256)]
    return np.concatenate(parts)


def reference_probabilities(record, frame, split, test_ids, digest):
    """Reconstruct the previously selected ensemble from verified prediction archives."""
    config = record["config"]
    validation = np.zeros((len(frame), len(CLASSES)))
    testing = np.zeros((len(test_ids), len(CLASSES)))
    for name, view, weight in zip(config["runs"], config["views"], record["selected"]["weights"], strict=True):
        if weight == 0:
            continue
        val, test, _ = load_component(name, frame, split, test_ids, digest, view)
        validation += weight * val
        testing += weight * test
    return validation, testing


def blend(vision, text, counts, config):
    """Reduce OCR influence in proportion to the amount of recognizable header text."""
    weight = config["ocr_weight"] * np.minimum(np.asarray(counts) / config["min_words_for_full_weight"], 1.0)
    return vision * (1 - weight[:, None]) + text * weight[:, None]


def paired_interval(delta, groups, replicates, seed):
    """Bootstrap the paired change in accuracy with exact-pixel duplicate groups kept together."""
    _, index = np.unique(groups, return_inverse=True)
    sizes = np.bincount(index)
    changes = np.bincount(index, weights=delta)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 50):
        sample = rng.integers(len(sizes), size=(min(50, replicates - start), len(sizes)))
        draws.extend(changes[sample].sum(1) / sizes[sample].sum(1))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def run():
    """Test one prespecified blend against the last submission and publish only if it passes."""
    config = json.loads(CONFIG.read_text())
    directory = data_directory()
    train, test, sample = read_metadata(directory)
    digest = metadata_digest(directory)
    ref = ROOT / "reports" / config["reference"]
    record = json.loads((ref / "selection.json").read_text())
    original = ROOT / record["submission"]
    if hashlib.sha256(original.read_bytes()).hexdigest() != record["submission_sha256"]:
        raise ValueError("Reference submission hash mismatch")
    val = pd.read_csv(ref / "validation_predictions.csv")
    split = pd.read_csv(ROOT / "reports/convnext_diversity/split.csv").sort_values("id").reset_index(drop=True)
    fit = train[train.id.isin(split.loc[split.split.eq("train"), "id"])].reset_index(drop=True)
    assert set(val.id) == set(split.loc[split.split.eq("validation"), "id"])
    assert not set(fit.id).intersection(val.id)
    assert train.set_index("id").loc[val.id, "label"].tolist() == val.label.tolist()
    assert train.set_index("id").loc[split.id, "label"].tolist() == split.label.tolist()
    headers, cache_key = read_headers(config, directory)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = ROOT / "checkpoint" / config["run_name"]
    report = ROOT / "reports" / config["run_name"]
    folder.mkdir(exist_ok=True)
    report.mkdir(exist_ok=True)
    train_texts = headers.loc[fit.id].tolist()
    train_text_hash = hashlib.sha256("\n".join(train_texts).encode()).hexdigest()
    signature = hashlib.sha256((cache_key + train_text_hash + CONFIG.read_text() + "\n".join(fit.id)).encode()).hexdigest()
    model, vocabs = train_head(train_texts, fit.label, config, device, folder, signature)
    vis, test_vis = reference_probabilities(record, val, split, test.id.tolist(), digest)
    prior = vis.argmax(1)
    if not np.array_equal(prior, val.prediction):
        raise ValueError("Reconstructed validation predictions differ from the incumbent")
    prior_test = pd.Series(test_vis.argmax(1), index=test.id)
    scored = pd.read_csv(original)
    validate_submission(scored, sample)
    if not np.array_equal(sample.id.map(prior_test), scored.label):
        raise ValueError("Reconstructed test labels differ from the incumbent submission")
    val_text = headers.loc[val.id]
    val_counts = val_text.map(word_count)
    p_ocr = probabilities(model, vocabs, val_text, device)
    mixed = blend(vis, p_ocr, val_counts, config)
    best = mixed.argmax(1)
    y = val.label.to_numpy()
    change = (best == y).astype(int) - (prior == y).astype(int)
    audit = pd.read_csv(ROOT / "data/cache_320/train_audit.csv").set_index("id").loc[val.id]
    assert audit.sha256.notna().all()
    ci = paired_interval(change, audit.sha256, config["bootstrap_replicates"], config["seed"])
    metrics = []
    for name, predictions in (("vision", prior), ("ocr", p_ocr.argmax(1)), ("vision_ocr", best)):
        metrics.append({"model": name, "accuracy": float(np.mean(y == predictions)),
                        "macro_f1": float(f1_score(y, predictions, labels=range(16), average="macro"))})
    passed = bool(ci[0] > 0 and change.sum() > 0 and metrics[2]["macro_f1"] > metrics[0]["macro_f1"])
    base_report, new_report = [classification_report(y, pred, labels=range(16), target_names=CLASSES,
                                output_dict=True, zero_division=0) for pred in (prior, best)]
    pd.DataFrame([{"class_name": c, "old_f1": base_report[c]["f1-score"], "new_f1": new_report[c]["f1-score"],
                   "delta_f1": new_report[c]["f1-score"] - base_report[c]["f1-score"]} for c in CLASSES]
                 ).to_csv(report / "class_comparison.csv", index=False)
    rows = val.copy()
    rows["old_prediction"], rows["prediction"], rows["ocr_words"] = prior, best, val_counts.to_numpy()
    rows["predicted_class"] = [CLASSES[i] for i in best]
    rows["confidence"] = mixed.max(1)
    rows.to_csv(report / "validation_predictions.csv", index=False)
    output, changed = original, 0
    if passed:
        test_text = headers.loc[test.id]
        test_probs = blend(test_vis, probabilities(model, vocabs, test_text, device), test_text.map(word_count), config)
        labels = sample.id.map(pd.Series(test_probs.argmax(1), index=test.id)).astype("int64")
        submission = pd.DataFrame({"id": sample.id, "label": labels})
        validate_submission(submission, sample)
        changed = int((submission.label != scored.label).sum())
        output = ROOT / "submissions/submission_ocr_header.csv"
        if output.exists() and not pd.read_csv(output).equals(submission):
            raise FileExistsError("Existing OCR submission differs")
        tmp = output.with_suffix(".partial.csv")
        submission.to_csv(tmp, index=False)
        validate_submission(pd.read_csv(tmp), sample)
        tmp.replace(output)
    summary = {"reference": config["reference"], "reference_sha256": record["submission_sha256"],
               "config": config, "metrics": metrics, "accepted": passed,
               "accuracy_gain": float(change.mean()), "paired_interval_95": ci,
               "corrected_errors": int((change == 1).sum()), "new_errors": int((change == -1).sum()),
               "validation_rows": len(y), "validation_nonempty": int(val_counts.gt(0).sum()),
               "validation_full_weight": int(val_counts.ge(config["min_words_for_full_weight"]).sum()),
               "test_rows": len(test), "test_changes": changed, "cache_key": cache_key,
               "training_text_sha256": train_text_hash, "ocr_text_sha256": hashlib.sha256("\n".join(headers).encode()).hexdigest(),
               "model_sha256": hashlib.sha256((folder / "head.pt").read_bytes()).hexdigest(),
               "submission": output.relative_to(ROOT).as_posix(), "submission_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "leaderboard_score": None, "device": str(device)}
    pd.DataFrame(metrics).to_csv(report / "comparison.csv", index=False)
    (report / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
