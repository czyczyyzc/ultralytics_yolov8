import unittest

from scripts.anti_uav.sweep_native_tracker_parameters import identity_counts


class TestIdentityDiagnostics(unittest.TestCase):
    def test_adjacent_correct_observations(self):
        frames = [[dict(id=i+1, box=[0, 0, 10, 10])] for i in range(2)]
        result = identity_counts(frames, {0: [[0, 0, 10, 10]], 1: [[0, 0, 10, 10]]}, {0, 1})
        self.assertEqual(result["continuous_gt_id_changes"], 1)
        self.assertEqual(result["adjacent_tp_frame_id_changes"], 1)

    def test_tracking_gap_is_not_adjacent(self):
        frames = [[dict(id=1, box=[0, 0, 10, 10])], [], [dict(id=2, box=[0, 0, 10, 10])]]
        gt = {i: [[0, 0, 10, 10]] for i in range(3)}
        result = identity_counts(frames, gt, {0, 1, 2})
        self.assertEqual(result["continuous_gt_id_changes"], 1)
        self.assertEqual(result["adjacent_tp_frame_id_changes"], 0)
        result = identity_counts(frames, gt, {0, 2})
        self.assertEqual(result["continuous_gt_id_changes"], 0)
        self.assertEqual(result["all_matched_observation_id_changes"], 1)

    def test_wrong_box_does_not_change_identity(self):
        frames = [[dict(id=1, box=[0, 0, 10, 10])], [dict(id=2, box=[20, 20, 30, 30])]]
        result = identity_counts(frames, {i: [[0, 0, 10, 10]] for i in range(2)}, {0, 1})
        self.assertEqual(result["all_matched_observation_id_changes"], 0)


if __name__ == "__main__":
    unittest.main()
