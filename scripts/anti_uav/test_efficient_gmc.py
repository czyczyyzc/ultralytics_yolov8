#!/usr/bin/env python3
import unittest
import cv2
import numpy as np
from efficient_gmc import EfficientGMC, SuppliedWarp


class TestGMC(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        cv2.setRNGSeed(20260917)

    def test_translation_coordinates(self):
        rng = np.random.default_rng(29)
        gray = rng.integers(0,256,(540,960),dtype=np.uint8)
        gray = cv2.GaussianBlur(gray,(5,5),1)
        frame = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        moved = cv2.warpAffine(frame,np.float32([[1,0,8],[0,1,4]]),(960,540))
        gmc = EfficientGMC(480,256,1)
        np.testing.assert_equal(gmc.apply(frame),np.eye(2,3))
        warp = gmc.apply(moved)
        np.testing.assert_allclose(warp[:,:2],np.eye(2),atol=.003)
        np.testing.assert_allclose(warp[:,2],[8,4],atol=.5)
        self.assertEqual(gmc.counts["estimated"],1)

    def test_blank_and_shape_change(self):
        gmc = EfficientGMC(320,128,5)
        for shape in ((540,960,3),(540,960,3),(240,320,3),(240,320,3)):
            np.testing.assert_equal(gmc.apply(np.zeros(shape,np.uint8)),np.eye(2,3))
        self.assertEqual(gmc.counts["identity_fallback"],3)

    def test_supplied_warp_once(self):
        holder = SuppliedWarp()
        with self.assertRaises(RuntimeError):
            holder.apply(None)
        holder.put(np.eye(2,3))
        with self.assertRaises(RuntimeError):
            holder.put(np.eye(2,3))
        np.testing.assert_equal(holder.apply(None),np.eye(2,3))
        with self.assertRaises(RuntimeError):
            holder.put(np.full((2,3),np.nan))


if __name__ == "__main__":
    unittest.main()
