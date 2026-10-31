"""Regression checks for image paths, label alignment, and duplicate-safe splits."""

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from scripts.model import DocumentDataset, build_model
from scripts.predict import validate_submission
from scripts.preprocess import ROOT, load_page, make_split, resolve_image
from scripts.preprocess import CLASSES
from scripts.train import validate_initial_checkpoint
from scripts.select_highres import aligned_probabilities
from scripts.select_ensemble import validate_candidates


class PipelineTests(unittest.TestCase):
    """Test the data contracts most likely to silently invalidate an experiment."""

    def setUp(self):
        """Keep temporary test files inside the writable project workspace."""
        (ROOT / ".cache").mkdir(exist_ok=True)

    def test_nested_paths_and_letterbox(self):
        """Find nested extraction layouts while retaining page proportions and margins."""
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            base = Path(directory)
            image_path = base / "train/train/form/page.tif"
            image_path.parent.mkdir(parents=True)
            Image.new("L", (20, 40), color=0).save(image_path)
            self.assertEqual(resolve_image(base, "train/form/page.tif"), image_path)
            array, dimensions, _, _ = load_page(image_path, 40)
            self.assertEqual(dimensions, (20, 40))
            self.assertTrue(np.all(array[:, :10] == 255))
            self.assertTrue(np.all(array[:, 10:30] == 0))
            with self.assertRaises(ValueError):
                resolve_image(base, "../outside.tif")

    def test_duplicate_groups_and_corrupt_rows(self):
        """Keep duplicate pages together and exclude unreadable rows from both splits."""
        labels = np.repeat(np.arange(16), 40)
        frame = pd.DataFrame({"id": [f"id_{i}" for i in range(len(labels))], "label": labels})
        audit = pd.DataFrame({"id": frame.id, "sha256": [f"hash_{i // 2}" for i in range(len(labels))],
                              "valid": True})
        audit.loc[0, "valid"] = False
        training, validation = make_split(frame, audit)
        self.assertNotIn(0, training)
        self.assertNotIn(0, validation)
        self.assertFalse(set(audit.iloc[training].sha256) & set(audit.iloc[validation].sha256))
        self.assertEqual(set(training) | set(validation), set(range(1, len(frame))))

    def test_submission_contract(self):
        """Reject shuffled IDs, fractional labels, and labels outside the class range."""
        sample = pd.DataFrame({"id": ["test_b", "test_a"], "label": [0, 0]})
        valid = pd.DataFrame({"id": sample.id, "label": [15, 3]})
        validate_submission(valid, sample)
        for invalid in (valid.iloc[::-1], valid.assign(label=[1.2, 3]), valid.assign(label=[16, 3])):
            with self.assertRaises(ValueError):
                validate_submission(invalid, sample)

    def test_training_gradient_and_cache_alignment(self):
        """Exercise an actual optimizer step and nonsequential cache row selection."""
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            path = Path(directory) / "images.npy"
            np.save(path, np.full((4, 32, 32), 255, dtype=np.uint8))
            dataset = DocumentDataset(path, [3, 1], labels=[0, 4, 2, 8])
            self.assertEqual(dataset[0][1], 8)
            model = build_model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            inputs = torch.stack([dataset[i][0] for i in range(2)])
            old = model.classifier[-1].weight.detach().clone()
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (2, 16))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([8, 4]))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(old, model.classifier[-1].weight))
            del dataset

    def test_docstrings(self):
        """Require a docstring for every repository Python class and function."""
        for folder in (ROOT / "scripts", ROOT / "tests"):
            for path in folder.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.assertIsNotNone(ast.get_docstring(node), f"Missing docstring: {path.name}:{node.lineno}")

    def test_warm_start_requires_identical_holdout(self):
        """Reject pretrained task checkpoints that saw any current validation image."""
        payload = {"classes": CLASSES, "metadata_digest": "dataset_a",
                   "config": {"model": "efficientnet_b0"}, "validation_ids": ["a", "b"]}
        validate_initial_checkpoint(payload, "efficientnet_b0", "dataset_a", ["b", "a"])
        with self.assertRaises(ValueError):
            validate_initial_checkpoint(payload, "efficientnet_b0", "dataset_a", ["a", "c"])
        with self.assertRaises(ValueError):
            validate_initial_checkpoint(payload, "efficientnet_b0", "dataset_b", ["a", "b"])
        with self.assertRaises(ValueError):
            validate_initial_checkpoint(payload, "small_cnn", "dataset_a", ["a", "b"])

    def test_geometry_is_seeded_and_validation_is_unchanged(self):
        """Exercise affine augmentation while keeping deterministic validation pixels."""
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            path = Path(directory) / "images.npy"
            pages = np.full((1, 64, 64), 255, dtype=np.uint8)
            pages[:, 10:40, 15:35] = 0
            np.save(path, pages)
            geometry = {"degrees": 1.5, "translate": 0.015, "scale": 0.03}
            training = DocumentDataset(path, [0], [7], "efficientnet_b0", True, geometry)
            plain = DocumentDataset(path, [0], [7], "efficientnet_b0")
            validation = DocumentDataset(path, [0], [7], "efficientnet_b0", False, geometry)
            torch.manual_seed(42)
            first, label = training[0]
            torch.manual_seed(42)
            second, _ = training[0]
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(label, 7)
            self.assertEqual(tuple(first.shape), (3, 64, 64))
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue(torch.equal(plain[0][0], validation[0][0]))
            for dataset in (training, plain, validation):
                dataset.images._mmap.close()
                dataset.images = None
            del training, plain, validation

    def test_probability_archives_align_ids_and_check_model_identity(self):
        """Prevent blends from mixing shuffled samples or stale checkpoint predictions."""
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            path = Path(directory) / "predictions.npz"
            probabilities = np.zeros((2, 16), dtype=np.float32)
            probabilities[0, 5] = 1
            probabilities[1, 2] = 1
            np.savez(path, ids=np.array(["a", "b"]), probabilities=probabilities,
                     checkpoint_sha256="expected_hash")
            actual = aligned_probabilities(path, ["b", "a"], "expected_hash")
            self.assertEqual(actual.argmax(axis=1).tolist(), [2, 5])
            with self.assertRaises(ValueError):
                aligned_probabilities(path, ["a", "c"], "expected_hash")
            with self.assertRaises(ValueError):
                aligned_probabilities(path, ["a", "b"], "wrong_hash")

    def test_ensemble_weights_define_probabilities(self):
        """Reject malformed mixtures before using validation results for selection."""
        validate_candidates([{"variant": "blend", "weights": [0.25, 0.25, 0.5]}], 3)
        for weights in ([0.5, 0.5], [-0.1, 0.6, 0.5], [0.2, 0.2, 0.2], [float("nan"), 0.0, 1.0]):
            with self.assertRaises(ValueError):
                validate_candidates([{"variant": "blend", "weights": weights}], 3)
        with self.assertRaises(ValueError):
            validate_candidates([{"variant": "same", "weights": [1]}, {"variant": "same", "weights": [1]}], 1)

    def test_backbones_receive_imagenet_normalized_three_channels(self):
        """Prevent a new backbone from silently receiving grayscale CNN normalization."""
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            path = Path(directory) / "pages.npy"
            np.save(path, np.full((1, 32, 32), 255, dtype=np.uint8))
            for name in ("efficientnet_b0", "resnet50", "convnext_tiny"):
                with self.subTest(backbone=name):
                    dataset = DocumentDataset(path, [0], model_name=name)
                    pixels, index = dataset[0]
                    self.assertEqual(tuple(pixels.shape), (3, 32, 32))
                    expected = (torch.ones(3) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
                    self.assertTrue(torch.allclose(pixels[:, 0, 0], expected))
                    self.assertEqual(index, 0)
                    dataset.images._mmap.close()
                    dataset.images = None


    def test_header_crop_precedes_padding(self):
        """Keep upper-page content separate from lower-page ink for a portrait scan."""
        from scripts.spatial_layout import PageViews
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            base = Path(directory)
            (base / "test").mkdir()
            page = np.zeros((200, 100), dtype=np.uint8)
            page[:80] = 255
            Image.fromarray(page).save(base / "test" / "page.tif")
            frame = pd.DataFrame({"filename": ["test/page.tif"]})
            dataset = PageViews(frame, base, {"page_wh": [40, 60], "header_wh": [60, 30], "header_fraction": 0.4})
            full, header = dataset[0]
            self.assertEqual(tuple(full.shape), (60, 40))
            self.assertEqual(tuple(header.shape), (30, 60))
            self.assertEqual(int(full.min()), 0)
            self.assertEqual(int(header.min()), 255)

    def test_shifted_predictions_use_white_borders_and_center_weight(self):
        """Check edge-sensitive probabilities for reversed white and black page IDs."""
        from scripts.shift_views import shifted_probabilities
        size = 80
        pages = np.stack([np.zeros((size, size), np.uint8), np.full((size, size), 255, np.uint8)])
        linear = torch.nn.Linear(3 * size * size, 16)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.fill_(-100)
            linear.bias[:2].zero_()
            linear.weight[0, 0], linear.weight[1, 0] = 10, -10
        model = torch.nn.Sequential(torch.nn.Flatten(), linear)
        center = np.eye(16, dtype=np.float32)[[0, 1]]
        probabilities = shifted_probabilities(model, pages, [1, 0], center, size, torch.device("cpu"))
        np.testing.assert_allclose(probabilities[:, 0], [1.0, 0.6], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
