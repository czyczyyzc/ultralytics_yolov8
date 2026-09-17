from collections import defaultdict
import unittest

from scripts.anti_uav.analyze_cached_tracker_suppression import evaluate, metrics


class SuppressionScoringTests(unittest.TestCase):
    def test_excluded_frames_do_not_enter_metrics(self):
        frames = [[dict(box=[0, 0, 10, 10])], [], [dict(box=[20, 20, 30, 30])]]
        gt = defaultdict(list, {0: [(0, 0, 10, 10)], 1: [(0, 0, 10, 10)]})
        total, per_frame = evaluate(frames, gt, {0, 2}, .5)
        self.assertEqual((total["tp"], total["fp"], total["fn"]), (1, 1, 0))
        self.assertNotIn(1, per_frame)

    def test_extra_box_is_false_positive(self):
        frames = [[dict(box=[0, 0, 10, 10]), dict(box=[0, 0, 10, 10])]]
        total, _ = evaluate(frames, {0: [(0, 0, 10, 10)]}, {0}, .5)
        self.assertEqual((total["tp"], total["fp"], total["fn"]), (1, 1, 0))

    def test_suppressed_true_box_reduces_recall(self):
        gt = {0: [(0, 0, 10, 10)]}
        det, _ = evaluate([[dict(box=[0, 0, 10, 10])]], gt, {0}, .5)
        track, _ = evaluate([[]], gt, {0}, .5)
        self.assertEqual(det["recall"], 1.)
        self.assertEqual(track["recall"], 0.)
        self.assertEqual(track["fn"], 1)

    def test_no_predictions_is_finite(self):
        values = metrics(dict(tp=0, fp=0, fn=1))
        self.assertEqual((values["precision"], values["recall"], values["f1"]), (0., 0., 0.))


if __name__ == "__main__":
    unittest.main()
