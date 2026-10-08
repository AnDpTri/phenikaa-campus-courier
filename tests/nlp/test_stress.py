from __future__ import annotations

import contextlib
import io
import random
import unittest
from unittest.mock import patch

from scripts.nlp import stress_nlp


class StressTests(unittest.TestCase):
    def test_strip_accents_preserves_case_and_vietnamese_d(self) -> None:
        self.assertEqual(stress_nlp.strip_accents("Đừng đến Thư Viện"), "Dung den Thu Vien")

    def test_punctuation_variant_retains_capitalisation(self) -> None:
        self.assertEqual(stress_nlp.strip_punctuation("Giao tới thư viện. [Không gấp!]"), "Giao tới thư viện Không gấp")

    def test_typos_are_reproducible(self) -> None:
        text = "giao toi thu vien phong hanh chinh"
        apply = stress_nlp.typos(0.5)
        self.assertEqual(apply(text, random.Random(123)), apply(text, random.Random(123)))
        self.assertEqual(stress_nlp.typos(0)(text, random.Random(123)), text)

    def test_stress_rejects_test_split_before_loading_any_data(self) -> None:
        for split in ("test", "train,test", "", "validation,unknown"):
            with self.subTest(split=split), patch("sys.argv", ["stress_nlp", "--splits", split]):
                with patch.object(stress_nlp, "load_dataset") as loader:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        stress_nlp.main()
                    self.assertEqual(error.exception.code, 2)
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
