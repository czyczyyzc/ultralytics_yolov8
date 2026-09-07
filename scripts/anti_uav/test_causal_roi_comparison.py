import unittest

from scripts.anti_uav.causal_roi_policy import choose_region, restore_boxes
from scripts.anti_uav.summarize_causal_roi_comparison import gt_boxes, score_rows, metrics, events


def track(box, tid=1, hits=3, score=.5, confirmed=True, predicted=False):
    return dict(box=box,id=tid,hits=hits,score=score,confirmed=confirmed,predicted=predicted)


class CausalRoiTests(unittest.TestCase):
    def test_acquisition_and_loss_require_full_frame(self):
        self.assertEqual(choose_region([],11,1920,1080)[1],"full_search")
        for t in (track([0,0,10,10],confirmed=False),track([0,0,10,10],predicted=True)):
            self.assertEqual(choose_region([t],11,1920,1080)[1],"full_search")

    def test_center_and_edge_keep_crop_size(self):
        t=track([950,530,970,550])
        self.assertEqual(choose_region([t],11,1920,1080)[0],(480,270,1440,810))
        edge=track([1900,1060,1920,1080])
        self.assertEqual(choose_region([edge],11,1920,1080,4)[0],(1440,810,1920,1080))

    def test_periodic_refresh_and_sticky_identity(self):
        a,b=track([20,20,40,40],1),track([950,530,970,550],2,100,.9)
        self.assertEqual(choose_region([a,b],11,1920,1080,previous_anchor=1)[2],1)
        self.assertEqual(choose_region([a,b],20,1920,1080)[1],"full_refresh")

    def test_translation_only_after_yolo_unletterbox(self):
        self.assertEqual(restore_boxes([[10,20,30,40,.8]],(480,270,1440,810),1920,1080),
                         [[490,290,510,310,.8]])

    def test_duplicate_and_absent_predictions_are_false_positive(self):
        gt=gt_boxes(dict(exist=[1,0,1],gt_rect=[[10,20,10,10],[0,0,0,0],[10,20,10,10]]))
        records=[dict(frame=0,detections=[[10,20,20,30,.8],[10,20,20,30,.7]]),
                 dict(frame=1,detections=[[0,0,10,10,.5]]),dict(frame=2,detections=[])]
        m=metrics(score_rows(records,gt,"detections"))
        self.assertEqual((m["tp"],m["fp"],m["fn"]),(1,2,1))

    def test_event_delay_and_fragment_count(self):
        rows=[dict(frame=i,visible=True,tp=int(i in (2,4,5)),id=1 if i<5 else 2) for i in range(6)]
        e=events(rows,100)[0]
        self.assertEqual((e["first_match_delay_frames"],e["fragments"],e["id_switches"]),(2,1,1))


if __name__ == "__main__":
    unittest.main()
