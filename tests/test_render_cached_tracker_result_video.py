import copy
import unittest

import numpy as np

from scripts.anti_uav.render_cached_tracker_result_video import crop_bounds, panel, validate_records


class CachedTrackerVideoTests(unittest.TestCase):
    def setUp(self):
        self.detections = [dict(frame_index=0, time_seconds=0,
                                boxes_xyxy_score=[[10, 20, 50, 60, .3]])]
        self.track = dict(id=9, box=[10, 20, 50, 60], score=.3,
                          confirmed=True, predicted=False, detection_index=0)
        self.rows = [dict(frame_index=0, time_seconds=0,
                          raw_tracks=[self.track], displayed_tracks=[self.track])]

    def test_exact_observation_and_empty(self):
        validate_records(self.detections, self.rows, 1, 100)
        validate_records([dict(frame_index=0, time_seconds=0, boxes_xyxy_score=[])],
                         [dict(frame_index=0, time_seconds=0, raw_tracks=[], displayed_tracks=[])], 1, 100)

    def test_reject_wrong_alignment(self):
        for key, value in (("frame_index", 1), ("time_seconds", .01)):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                validate_records(self.detections, rows, 1, 100)
        with self.assertRaises(ValueError):
            validate_records(self.detections, [], 1, 100)

    def test_reject_fabricated_or_duplicate_output(self):
        for key, value in (("confirmed", False), ("predicted", True), ("id", -1),
                           ("detection_index", -1), ("box", [11, 20, 51, 60]), ("score", .9)):
            rows = copy.deepcopy(self.rows)
            rows[0]["displayed_tracks"][0][key] = value
            with self.assertRaises(ValueError):
                validate_records(self.detections, rows, 1, 100)
        rows = copy.deepcopy(self.rows)
        rows[0]["displayed_tracks"].append(rows[0]["displayed_tracks"][0])
        with self.assertRaises(ValueError):
            validate_records(self.detections, rows, 1, 100)

    def test_panel_does_not_mutate_source_or_tracks(self):
        frame = np.full((1080, 1920, 3), 128, np.uint8)
        before, metadata = frame.copy(), copy.deepcopy(self.track)
        result = panel(frame, [self.track], 0, 1, 100, "Dist public-code + GMC")
        self.assertEqual(result.shape, (784, 1600, 3))
        np.testing.assert_array_equal(frame, before)
        self.assertEqual(self.track, metadata)
        empty = panel(frame, [], 0, 1, 100, "Dist public-code + GMC")
        np.testing.assert_array_equal(empty[64:, :1280], 128)

    def test_crop_bounds_keep_small_large_edge_targets(self):
        for box in ([0, 0, 4, 6], [600, 600, 900, 850], [0, 0, 1920, 1080]):
            x, y, w, h = crop_bounds(box, 1920, 1080)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x+w, 1920)
            self.assertLessEqual(y+h, 1080)
            self.assertLessEqual(x, box[0])
            self.assertLessEqual(y, box[1])
            self.assertGreaterEqual(x+w, box[2])
            self.assertGreaterEqual(y+h, box[3])


if __name__ == "__main__":
    unittest.main()
