from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from courier.nlp import MissionParser, TargetSpec
from courier.nlp.neural import (
    HybridMissionParser,
    HybridThresholds,
    NeuralMissionParser,
    NeuralParserNet,
    _decode,
    encode_texts,
    spec_labels,
)
from courier.nlp.synth import GOAL_MODES, TYPES, generate, spec_from_mission

DATA = Path("Phenikaa_Campus_Courier_2026_v3/delivery_public")


class SynthTests(unittest.TestCase):
    def test_generation_is_reproducible_and_labels_are_valid(self) -> None:
        first, second = generate(200, seed=7), generate(200, seed=7)
        self.assertEqual(first, second)
        for example in first:
            goal = example.goal
            self.assertIn(goal.ref or "named", GOAL_MODES)
            map_only = goal.ref == "anchor_near" or (goal.ref or "").endswith("_most")
            self.assertEqual(goal.type is None, map_only)
            if goal.ref in ("near", "far", "anchor_near"):
                self.assertIn(goal.anchor, TYPES)
                self.assertNotEqual(goal.anchor, goal.type)
            if example.via is not None:
                self.assertIn(example.via.type, TYPES)
                self.assertNotEqual(example.via.type, goal.type)
            spec_labels(example.goal, example.via, example.urgent, example.fragile)

    def test_holdout_uses_different_core_phrasing(self) -> None:
        train_texts = " ".join(e.text.lower() for e in generate(300, seed=1))
        holdout_texts = " ".join(e.text.lower() for e in generate(300, seed=1, holdout=True))
        self.assertNotEqual(train_texts, holdout_texts)

    @unittest.skipUnless((DATA / "train" / "scenes.json").exists(), "dataset not available")
    def test_annotation_labels_match_the_rule_parser_on_validation(self) -> None:
        from courier.common import load_dataset

        parser = MissionParser()
        dataset = load_dataset(DATA, "validation")
        agree = sum(parser.parse(s.mission.text).goal == spec_from_mission(s.mission)[0] for s in dataset.scenes)
        self.assertGreater(agree / len(dataset.scenes), 0.95)


class NeuralTests(unittest.TestCase):
    def test_encoding_shape_and_padding(self) -> None:
        pieces = encode_texts(["Giao tới thư viện.", "a"])
        self.assertEqual(pieces.shape[0], 2)
        self.assertTrue((pieces[1, 1:] == 0).all())

    def test_decode(self) -> None:
        self.assertEqual(_decode("goal", "named", 1, 0), TargetSpec(TYPES[0]))
        self.assertEqual(_decode("goal", "anchor_near", 3, 2), TargetSpec(None, "anchor_near", TYPES[1]))
        self.assertEqual(_decode("goal", "west_most", 3, 2), TargetSpec(None, "west_most"))
        self.assertEqual(_decode("goal", "near", 1, 2), TargetSpec(TYPES[0], "near", TYPES[1]))
        self.assertIsNone(_decode("via", "none", 1, 2))

    def test_artifact_round_trip_and_hybrid_defaults_to_rules(self) -> None:
        torch.manual_seed(0)
        model = NeuralMissionParser(NeuralParserNet(dim=8, hidden=8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            model.save(path, thresholds={"goal": 2.0, "via": 2.0, "flags": 2.0})
            hybrid = HybridMissionParser.load(path)
        self.assertEqual(hybrid.thresholds, HybridThresholds(2.0, 2.0, 2.0))
        rules = MissionParser()
        for text in ("Giao tới thư viện, hàng dễ vỡ.", "Ghé căn tin trước rồi mang tới ký túc xá phía bắc."):
            self.assertEqual(hybrid.parse(text), rules.parse(text))

    def test_hybrid_fills_missing_goal(self) -> None:
        model = NeuralMissionParser(NeuralParserNet(dim=8, hidden=8))
        hybrid = HybridMissionParser(model, HybridThresholds(2.0, 2.0, 2.0))
        rule = hybrid.rules.parse("xin chao")
        prediction = model.predict("xin chao")
        combined = hybrid.combine(rule, prediction)
        self.assertEqual(combined.goal, prediction.parsed.goal)


class MapAwareTests(unittest.TestCase):
    def test_goal_absent_from_map_is_replaced_by_a_present_type(self) -> None:
        torch.manual_seed(0)
        hybrid = HybridMissionParser(NeuralMissionParser(NeuralParserNet(dim=8, hidden=8)), HybridThresholds(2.0, 2.0, 2.0))
        landmarks = (("canteen", (0, 0)), ("dorm", (1, 2)))
        present = hybrid.parse_for_map("Giao tới căn tin", landmarks)
        self.assertEqual(present.goal, TargetSpec("canteen"))
        absent = hybrid.parse_for_map("Giao tới thư viện", landmarks)
        self.assertIn(absent.goal.type, {"canteen", "dorm"})
        self.assertTrue(hybrid.last_sources["goal"].startswith("neural_map"))


class AugmentRealTests(unittest.TestCase):
    def test_names_change_within_type_and_items_are_left_alone(self) -> None:
        import random

        from courier.nlp.synth import augment_real

        text = "Mang the thu vien den thu vien giup minh. Dung nham voi can tin."
        outputs = {augment_real(text, random.Random(seed), rate=1.0) for seed in range(20)}
        self.assertGreater(len(outputs), 1)
        parser = MissionParser()
        for out in outputs:
            self.assertIn("the thu vien", out)
            self.assertNotIn("can tin", out.split(".")[0])
        self.assertTrue(any(parser.parse(out).goal == TargetSpec("library") for out in outputs))


if __name__ == "__main__":
    unittest.main()
