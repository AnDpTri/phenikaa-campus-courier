from __future__ import annotations

import unittest
from pathlib import Path

from courier.common import load_dataset
from courier.nlp import MissionParser, TargetSpec, parse_mission, resolve, resolve_target
from courier.nlp.text import Speller, fold, tokenize, typo_sources

DATA_ROOT = Path(__file__).parents[2] / "Phenikaa_Campus_Courier_2026_v3" / "delivery_public"


class TextTests(unittest.TestCase):
    def test_fold_strips_vietnamese_accents(self) -> None:
        self.assertEqual(fold("Đừng nhầm với Ký Túc Xá"), "dung nham voi ky tuc xa")
        self.assertEqual(tokenize("Hỏa tốc! (gấp)"), ["hoa", "toc", "!", "(", "gap", ")"])

    def test_typo_sources_cover_swaps(self) -> None:
        self.assertIn("nham", typo_sources("nhma"))

    def test_speller_repairs_drops_and_swaps(self) -> None:
        speller = Speller({"khong": 100, "phong": 100, "nham": 50, "voi": 50, "dung": 50}, {"dung nham": 40, "nham voi": 40})
        self.assertEqual(speller(["kong", "phng"]), ["khong", "phong"])
        self.assertEqual(speller(["dung", "hnam", "voi"]), ["dung", "nham", "voi"])


class ParserTests(unittest.TestCase):
    def test_plain_goal_with_and_without_accents(self) -> None:
        for text in ("Mang hộp giấy tới thư viện giúp mình.", "mang hop giay toi thu vien giup minh"):
            parsed = parse_mission(text)
            self.assertEqual(parsed.goal, TargetSpec("library"))
            self.assertIsNone(parsed.via)

    def test_distractors_and_corrections_are_ignored(self) -> None:
        parsed = parse_mission(
            "Lúc nãy nhắn nhầm là căn tin. Đúng ra: giao tới không phải nhà thi đấu mà là trạm y tế. "
            "Đừng nhầm với bãi xe nhé. Người nhận đã rời thư viện rồi."
        )
        self.assertEqual(parsed.goal, TargetSpec("clinic"))
        self.assertIsNone(parsed.via)

    def test_via_before_and_after_goal(self) -> None:
        first = parse_mission("Ghé căn tin lấy khay cơm trước, rồi mang bưu kiện tới ký túc xá.")
        later = parse_mission("Trước khi đưa bưu kiện tới ký túc xá, nhớ ghé căn tin lấy khay cơm.")
        pickup = parse_mission("Lấy khay cơm ở căn tin xong thì mang bưu kiện tới ký túc xá.")
        for parsed in (first, later, pickup):
            self.assertEqual(parsed.goal, TargetSpec("dorm"))
            self.assertEqual(parsed.via, TargetSpec("canteen"))

    def test_spatial_references(self) -> None:
        self.assertEqual(parse_mission("Giao tới căn tin phía bắc.").goal, TargetSpec("canteen", "north"))
        self.assertEqual(
            parse_mission("Giao tới căn tin gần thư viện hơn.").goal, TargetSpec("canteen", "near", "library")
        )
        self.assertEqual(
            parse_mission("Giao tới căn tin nằm xa cổng trường.").goal, TargetSpec("canteen", "far", "gate")
        )

    def test_map_only_descriptions(self) -> None:
        self.assertEqual(
            parse_mission("Giao tới địa điểm nằm cao nhất trên bản đồ.").goal, TargetSpec(None, "north_most")
        )
        self.assertEqual(parse_mission("Đích đến là chỗ ở rìa trái nhất.").goal, TargetSpec(None, "west_most"))
        self.assertEqual(
            parse_mission("Cần túi đồ ở nơi cạnh thư viện nhất.").goal, TargetSpec(None, "anchor_near", "library")
        )

    def test_function_descriptions_of_places(self) -> None:
        self.assertEqual(parse_mission("Giao hồ sơ tới nơi nộp hồ sơ.").goal, TargetSpec("office"))
        self.assertEqual(parse_mission("Mang thuốc tới nơi khám sức khỏe.").goal, TargetSpec("clinic"))

    def test_urgency_and_fragility_with_negation(self) -> None:
        self.assertTrue(parse_mission("Giao tới thư viện. Hỏa tốc!").urgent)
        self.assertTrue(parse_mission("Giao tới thư viện. Không được chậm trễ.").urgent)
        self.assertFalse(parse_mission("Giao tới thư viện. Không gấp đâu.").urgent)
        self.assertTrue(parse_mission("Giao tới thư viện. Hàng dễ vỡ.").fragile)
        self.assertFalse(parse_mission("Giao tới thư viện. Hàng bền, không lo vỡ.").fragile)
        self.assertFalse(parse_mission("Giao tới thư viện. Đồ không vỡ được đâu.").fragile)

    def test_flag_phrases_tolerate_real_word_typos(self) -> None:
        self.assertTrue(parse_mission("Giao tới thư viện. Không được cậm trễ.").urgent)
        self.assertFalse(parse_mission("Giao tới thư viện. Rơi cũng cẳng sao.").fragile)

    def test_parser_accepts_explicit_vocabulary(self) -> None:
        parsed = MissionParser({"unigrams": {}, "bigrams": {}}).parse("giao toi thu vien")
        self.assertEqual(parsed.goal, TargetSpec("library"))


class ResolverTests(unittest.TestCase):
    LANDMARKS = (("canteen", (0, 1)), ("canteen", (4, 3)), ("library", (3, 3)), ("gate", (2, 0)))

    def test_directional_copy(self) -> None:
        kind, ref = resolve_target(TargetSpec("canteen", "north"), self.LANDMARKS)
        self.assertEqual((kind, ref.rc), ("canteen", (0, 1)))

    def test_near_far_use_straight_line_distance(self) -> None:
        _, near = resolve_target(TargetSpec("canteen", "near", "library"), self.LANDMARKS)
        _, far = resolve_target(TargetSpec("canteen", "far", "library"), self.LANDMARKS)
        self.assertEqual((near.rc, far.rc), ((4, 3), (0, 1)))

    def test_map_only_targets_pick_any_type(self) -> None:
        self.assertEqual(resolve_target(TargetSpec(None, "west_most"), self.LANDMARKS)[0], "gate")
        kind, ref = resolve_target(TargetSpec(None, "anchor_near", "library"), self.LANDMARKS)
        self.assertEqual((kind, ref.rc, ref.anchor), ("canteen", (4, 3), "library"))

    def test_single_copy_needs_no_ref(self) -> None:
        self.assertEqual(resolve_target(TargetSpec("gate", "north"), self.LANDMARKS), ("gate", None))

    def test_resolve_never_returns_a_missing_goal(self) -> None:
        mission = resolve(parse_mission("Giao tới phòng thí nghiệm."), self.LANDMARKS)
        self.assertIn(mission.goal, {kind for kind, _ in self.LANDMARKS})


@unittest.skipUnless((DATA_ROOT / "validation" / "scenes.json").exists(), "dataset not available")
class AnnotatedMissionTests(unittest.TestCase):
    def test_validation_missions_resolve_to_annotated_targets(self) -> None:
        dataset = load_dataset(DATA_ROOT, "validation")
        assert dataset.scenes is not None
        parser = MissionParser()
        correct = 0
        for scene in dataset.scenes:
            mission = resolve(parser.parse(scene.mission.text), scene.landmarks)
            truth = scene.mission
            correct += int(
                scene.landmark_candidates(mission.goal, mission.goal_ref)
                == scene.landmark_candidates(truth.goal, truth.goal_ref)
                and (mission.via is None) == (truth.via is None)
                and (
                    truth.via is None
                    or scene.landmark_candidates(mission.via, mission.via_ref)
                    == scene.landmark_candidates(truth.via, truth.via_ref)
                )
                and mission.urgent == truth.urgent
                and mission.fragile == truth.fragile
            )
        self.assertGreaterEqual(correct / len(dataset.scenes), 0.97)


if __name__ == "__main__":
    unittest.main()
