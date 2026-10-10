"""Score changes must use the scored subset size, not the full test size."""
import unittest

from scripts.compare_submissions import score_bound


class ScoreBoundTests(unittest.TestCase):
    def test_changes_can_concentrate_on_public_subset(self):
        reference = [0] * 100
        candidate = [1] * 10 + [0] * 90
        self.assertEqual(score_bound(candidate, reference, 10), 10.0)
        self.assertEqual(score_bound(candidate, reference, 1), 100.0)

    def test_unknown_subset_bound_uses_most_changed_scenes(self):
        reference = [0] * 30
        candidate = [1] * 8 + [0] * 2 + [1] * 2 + [0] * 18
        self.assertEqual(score_bound(candidate, reference, 2), 50.0)

    def test_bad_scene_counts_are_rejected(self):
        for count in (0, 3):
            with self.assertRaises(ValueError):
                score_bound([0] * 20, [0] * 20, count)


if __name__ == "__main__":
    unittest.main()
