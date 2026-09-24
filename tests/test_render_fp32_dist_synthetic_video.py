import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.anti_uav.render_fp32_dist_synthetic_video import (
    iter_cache, manifest_rows, selected_frame,
)


class SyntheticVideoSourceTest(unittest.TestCase):
    def test_replacement_only_changes_approved_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = np.full((8, 10, 3), 12, np.uint8)
            replacement = np.full((8, 10, 3), 32, np.uint8)
            import cv2
            cv2.imwrite(str(root / "new.png"), replacement)
            rows = {4: dict(frame=4, state="replaced", replacement="new.png"),
                    5: dict(frame=5, state="original_retained")}
            self.assertTrue(np.array_equal(selected_frame(image, root, rows, 4, "real"), image))
            self.assertTrue(np.array_equal(selected_frame(image, root, rows, 4, "synthetic"), replacement))
            self.assertTrue(np.array_equal(selected_frame(image, root, rows, 5, "synthetic"), image))
            self.assertTrue(np.array_equal(selected_frame(image, root, rows, 6, "synthetic"), image))

    def test_manifest_requires_complete_video_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps(dict(
                training_allowed=False, rows=[dict(video="Video00009", frame=0, state="replaced")]
            )))
            with self.assertRaises(ValueError):
                manifest_rows(root)

    def test_cache_rejects_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.jsonl"
            path.write_text(json.dumps(dict(frame_index=1, time_seconds=.01)) + "\n")
            with self.assertRaises(ValueError):
                list(iter_cache(path, 2))


if __name__ == "__main__":
    unittest.main()
