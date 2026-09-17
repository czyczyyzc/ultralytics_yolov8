import unittest
from scripts.anti_uav.render_tracker_forensic_clips import display_detections


class DetectionDisplayTests(unittest.TestCase):
    def test_unassociated_detection_is_not_hidden(self):
        result = display_detections([[1,2,3,4,.04]],[])
        self.assertEqual(result[0]["box"],[1,2,3,4])
        self.assertIsNone(result[0]["id"])
        self.assertEqual(result[0]["state"],"pending_detection")

    def test_id_only_for_confirmed_observation(self):
        t=dict(id=4,box=[1,2,3,4],score=.5,confirmed=False,predicted=False)
        self.assertIsNone(display_detections([[1,2,3,4,.5]],[t])[0]["id"])
        t["confirmed"]=True
        self.assertEqual(display_detections([[1,2,3,4,.5]],[t])[0]["id"],4)
        t["predicted"]=True
        self.assertIsNone(display_detections([[1,2,3,4,.5]],[t])[0]["id"])

    def test_extra_tracks_do_not_invent_detections(self):
        t=dict(id=4,box=[1,2,3,4],score=.5,confirmed=True,predicted=False)
        self.assertEqual(display_detections([],[t]),[])


if __name__ == "__main__":
    unittest.main()
