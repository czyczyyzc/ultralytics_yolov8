import unittest

from scripts.anti_uav.audit_native_tracker_association import validate_supplement
from scripts.anti_uav.cache_tracker_low_score_detections import merge_low_detections


class TestLowScoreCache(unittest.TestCase):
    def test_only_low_boxes_are_added(self):
        high = [[1, 2, 4, 5, .4]]
        combined, added = merge_low_detections(high, high + [[6, 7, 8, 9, .02]], .01, .03)
        self.assertEqual(added, 1)
        self.assertEqual(combined.shape, (2, 5))

    def test_changed_high_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_low_detections([[1, 2, 4, 5, .4]], [[1, 2, 4, 5, .3]], .01, .03)

    def test_empty_original(self):
        boxes, added = merge_low_detections([], [[6, 7, 8, 9, .02]], .01, .03)
        self.assertEqual(added, 1)
        self.assertEqual(len(boxes), 1)

    def test_replay_integrity(self):
        original = [dict(frame_index=0, time_seconds=0., boxes_xyxy_score=[[1, 2, 4, 5, .4]])]
        extra = [dict(original[0], boxes_xyxy_score=[[1, 2, 4, 5, .4], [6, 7, 8, 9, .02]])]
        self.assertEqual(validate_supplement(original, extra), 1)
        extra[0]["boxes_xyxy_score"][0][0] += 1
        with self.assertRaises((ValueError, AssertionError)):
            validate_supplement(original, extra)


if __name__ == "__main__":
    unittest.main()
