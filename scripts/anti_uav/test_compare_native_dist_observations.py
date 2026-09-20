import io
import json
import unittest
from compare_native_dist_observations import compare


def stream(count=3, change=None):
    rows = [dict(frame_index=i, boxes_xyxy_score=[], displayed_tracks=[],
                 warp=[[1, 0, 0], [0, 1, 0]]) for i in range(count)]
    if change:
        change(rows)
    return io.StringIO("".join(json.dumps(row) + "\n" for row in rows))


class ComparisonTests(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(compare(stream(), stream())["passed"])

    def test_differences(self):
        def change(rows):
            rows[1]["displayed_tracks"] = [dict(id=2)]
        result = compare(stream(), stream(change=change))
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_differing_frames"]["displayed_tracks"], [1])

    def test_missing_frame(self):
        with self.assertRaises(ValueError):
            compare(stream(), stream(2))

    def test_explicit_prefix(self):
        self.assertTrue(compare(stream(5), stream(3), 3)["passed"])

    def test_excess_prefix_candidate(self):
        with self.assertRaises(ValueError):
            compare(stream(5), stream(4), 3)

    def test_short_prefix(self):
        with self.assertRaises(ValueError):
            compare(stream(2), stream(2), 3)

    def test_empty(self):
        with self.assertRaises(ValueError):
            compare(stream(0), stream(0))

    def test_order(self):
        with self.assertRaises(ValueError):
            compare(stream(), stream(change=lambda rows: rows.reverse()))


if __name__ == "__main__":
    unittest.main()
