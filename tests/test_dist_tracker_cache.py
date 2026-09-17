import unittest
from types import SimpleNamespace
import numpy as np

from scripts.anti_uav.evaluate_dist_tracker_cache import as_results, recover_observation, CachedGMC


class DistTrackerCacheTests(unittest.TestCase):
    def test_coordinates_and_empty(self):
        r = as_results([[10,20,14,26,.03]])
        np.testing.assert_array_equal(r.xywh, [[12,23,4,6]])
        self.assertEqual(as_results([]).xywh.shape, (0,4))
        with self.assertRaises(ValueError):
            as_results([[1,2,0,4,.5]])

    def test_observation_is_not_kalman_box(self):
        t = SimpleNamespace(frame_id=5,idx=0,score=.6,track_id=9,is_activated=True,
                            xyxy=np.array([11,20,31,40]))
        r = recover_observation(t,[[10,20,30,40,.6]],5)
        self.assertEqual(r["box"],[10,20,30,40])
        self.assertNotEqual(r["box"],r["kalman_box"])
        with self.assertRaises(ValueError):
            recover_observation(t,[[10,20,30,40,.6]],6)

    def test_gmc_advance_and_copy(self):
        warps = np.array([np.eye(2,3),np.eye(2,3)])
        g = CachedGMC(warps)
        g.apply(None)[0,0] = 10
        self.assertEqual(warps[0,0,0],1)
        g.apply(None)
        with self.assertRaises(ValueError):
            g.apply(None)


if __name__ == "__main__":
    unittest.main()
