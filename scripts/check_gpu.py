"""Verify CUDA training and estimate throughput before a full fine-tuning run."""

import json
import time

import torch

from scripts.model import build_model
from scripts.preprocess import ROOT


def check_gpu():
    """Run AMP optimizer steps at the candidate shape and record memory/throughput."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected Python environment")
    config = json.loads((ROOT / "configs/efficientnet_improved.json").read_text())
    torch.set_num_threads(config["threads"])
    torch.hub.set_dir(str(ROOT / "checkpoint/pretrained"))
    model = build_model(config["model"], pretrained=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    images = torch.rand(config["batch_size"], 3, config["image_size"], config["image_size"], device="cuda")
    targets = torch.arange(config["batch_size"], device="cuda") % 16
    durations = []
    for step in range(12):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if step >= 2:
            durations.append(time.perf_counter() - started)
    result = {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "batch_size": config["batch_size"],
              "image_size": config["image_size"], "finite_loss": bool(torch.isfinite(loss).item()),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
              "synthetic_training_images_per_second": config["batch_size"] * len(durations) / sum(durations),
              "note": "Synthetic GPU batch only; excludes TIFF/cache loading and validation"}
    if not result["finite_loss"]:
        raise RuntimeError("Nonfinite loss in CUDA smoke check")
    (ROOT / "reports/gpu_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    check_gpu()
