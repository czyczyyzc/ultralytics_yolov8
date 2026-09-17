import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.anti_uav.render_cached_rk_botsort_video import confirmed_observed, load_records
from scripts.anti_uav.render_pt_detector_video import panel


class CachedTrackerVideoTests(unittest.TestCase):
    def records(self, rows, count):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return load_records(path, count, 100.)

    def test_empty_and_nonempty_frames(self):
        records = self.records([
            dict(frame_index=0, time_seconds=0., boxes_xyxy_score=[]),
            dict(frame_index=1, time_seconds=.01, boxes_xyxy_score=[[1, 2, 4, 6, .5]])], 2)
        self.assertEqual(records[0].shape, (0, 5))
        self.assertEqual(records[1].shape, (1, 5))

    def test_reject_gap_timestamp_count_and_invalid_boxes(self):
        valid = dict(frame_index=0, time_seconds=0., boxes_xyxy_score=[])
        for change in (dict(frame_index=1), dict(time_seconds=.1),
                       dict(boxes_xyxy_score=[[1, 2, 0, 6, .5]]),
                       dict(boxes_xyxy_score=[[1, 2, 4, 6, float("nan")]])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.records([dict(valid, **change)], 1)
        with self.assertRaises(ValueError):
            self.records([valid], 2)

    def test_confirmation_and_prediction_filters(self):
        tracks = [dict(id=1, confirmed=False, predicted=False, score=.9),
                  dict(id=2, confirmed=True, predicted=True, score=.8),
                  dict(id=3, confirmed=True, predicted=False, score=.4),
                  dict(id=4, confirmed=True, predicted=False, score=.7)]
        self.assertEqual([t["id"] for t in confirmed_observed(tracks)], [4, 3])

    def test_panels_preserve_clean_source(self):
        source = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        original = source.copy()
        for boxes, tracks in ((np.empty((0, 5), np.float32), []),
                              (np.array([[0, 0, 8, 8, .3], [1908, 1066, 1919, 1079, .2]], np.float32),
                               [dict(id=1), dict(id=2)])):
            result = panel(source, boxes, 0, 10, 100., "Tracker test", tracks=tracks)
            self.assertEqual(result.shape, (784, 1600, 3))
            np.testing.assert_array_equal(source, original)
        with self.assertRaises(ValueError):
            panel(source, np.empty((0, 5), np.float32), 0, 10, 100., "test", tracks=[dict(id=1)])


if __name__ == "__main__":
    unittest.main()
