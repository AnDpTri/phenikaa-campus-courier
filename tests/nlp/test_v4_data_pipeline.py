from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "v4_data_pipeline", ROOT / "scripts" / "nlp" / "v4_data_pipeline.py"
)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class V4DataPipelineTests(unittest.TestCase):
    def test_valid_partial_via_candidate(self) -> None:
        document = {
            "version": 1,
            "batch_id": "trial",
            "banks": {
                "VIA_FRAMES_BEFORE": ["tien duong ghe {W} nhan {I2}, sau do hay"],
            },
        }
        problems, counts = pipeline.validate_document(document, partial=True)
        self.assertEqual(problems, [])
        self.assertEqual(counts["VIA_FRAMES_BEFORE"], 1)

    def test_rejects_accent_and_wrong_placeholder(self) -> None:
        document = {
            "version": 1,
            "batch_id": "bad",
            "banks": {
                "VIA_FRAMES_AFTER": ["nhớ ghé {L} trước"],
            },
        }
        problems, _ = pipeline.validate_document(document, partial=True)
        messages = " ".join(problem.message for problem in problems)
        self.assertIn("lowercase ASCII", messages)
        self.assertIn("unexpected placeholders", messages)
        self.assertIn("missing placeholders", messages)

    def test_rejects_duplicate_existing_phrase(self) -> None:
        document = {
            "version": 1,
            "batch_id": "duplicate",
            "banks": {
                "VIA_FRAMES_AFTER": [pipeline.synth.VIA_FRAMES_AFTER[0]],
            },
        }
        problems, _ = pipeline.validate_document(document, partial=True)
        self.assertTrue(any("duplicates an existing phrase" in item.message for item in problems))

    def test_rejects_wrong_direction_semantics(self) -> None:
        document = {
            "version": 1,
            "batch_id": "direction",
            "banks": {"MOST_PHRASES": {"north_most": ["o duoi cung ban do"]}},
        }
        problems, _ = pipeline.validate_document(document, partial=True)
        self.assertTrue(any("north_most" in item.message for item in problems))

    def test_rejects_alias_already_present_in_base_lexicon(self) -> None:
        document = {
            "version": 1,
            "batch_id": "base-alias-duplicate",
            "banks": {"EXTRA_PLACES": {"library": ["thu vien"]}},
        }
        problems, _ = pipeline.validate_document(document, partial=True)
        self.assertTrue(any("duplicates an existing phrase" in item.message for item in problems))

    def test_rejects_extreme_phrase_already_present_in_base_lexicon(self) -> None:
        document = {
            "version": 1,
            "batch_id": "base-extreme-duplicate",
            "banks": {"MOST_PHRASES": {"north_most": ["ria tren nhat"]}},
        }
        problems, _ = pipeline.validate_document(document, partial=True)
        self.assertTrue(any("duplicates an existing phrase" in item.message for item in problems))


if __name__ == "__main__":
    unittest.main()
