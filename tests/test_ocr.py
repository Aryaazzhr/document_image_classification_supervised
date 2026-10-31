"""Check the OCR branch's ID ordering, empty headers, and confidence gate."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import torch

from scripts.extract_ocr import read_chunk
from scripts.ocr_classifier import HeaderClassifier, blend, clean_header, tensors, word_count


class HeaderTests(unittest.TestCase):
    """Cover low-text scans and sparse batch invariants that affect the final predictions."""

    def test_empty_and_numeric_headers_keep_visual_predictions(self):
        """Only recognizable words should grant the text model any influence."""
        visual = np.array([[0.8, 0.2]] * 3)
        text = 1 - visual
        headers = ["", "9013872 -- 123", "invoice amount " * 12]
        counts = [word_count(clean_header(header)) for header in headers]
        result = blend(visual, text, counts, {"ocr_weight": 0.38, "min_words_for_full_weight": 12})
        np.testing.assert_array_equal(result[:2], visual[:2])
        np.testing.assert_allclose(result[2], [0.572, 0.428])

    def test_blank_sparse_row_keeps_position_and_finite_gradients(self):
        """A document with no recognized vocabulary must not shift the next row's logits."""
        matrix = csr_matrix(np.array([[0, 1, 0], [0, 0, 0], [0.7, 0, 0.2]], dtype=np.float32))
        model = HeaderClassifier(3, 8).eval()
        batch = model(*tensors(matrix, "cpu"))
        for i in range(3):
            torch.testing.assert_close(batch[i:i+1], model(*tensors(matrix[i], "cpu")))
        batch.sum().backward()
        self.assertTrue(torch.isfinite(model.terms.weight.grad).all())

    def test_cache_retains_empty_text_and_rejects_reordered_ids(self):
        """A blank header remains a valid row and IDs must match its source batch."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "chunk.tsv"
            path.write_text("a\t\nb\tSubject: test\n", encoding="utf-8")
            self.assertEqual(read_chunk(path, ["a", "b"]), ["", "Subject: test"])
            with self.assertRaises(ValueError):
                read_chunk(path, ["b", "a"])
