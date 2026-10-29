from scripts.anti_uav.prepare_labeler_training_expansion import frame_plan


def row(index, status="confirmed", box=(0, 0, 100, 100)):
    return dict(frame_index=index, status=status,
                **dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), box)))


def test_pending_absent_and_unknown_are_not_negatives():
    plan, excluded = frame_plan([row(0), row(1, "absent"), row(2, "uncertain")], 4, 100, 100, False)
    assert plan == {0: [[.5, .5, 1., 1.]]}
    assert excluded == dict(unlabelled=1, unreviewed_absent=1, uncertain=1)


def test_reviewed_negative_only_video_is_allowed():
    plan, excluded = frame_plan([row(0, "absent")], 1, 100, 100, True)
    assert plan == {0: []}
    assert excluded == dict(unlabelled=0)


def test_uncertain_object_excludes_whole_frame():
    plan, excluded = frame_plan([row(0), row(0, "uncertain")], 1, 100, 100, True)
    assert not plan
    assert excluded["uncertain"] == 1


def test_no_tiny_or_large_area_filter():
    plan, _ = frame_plan([row(0, box=(1, 1, 5, 5)), row(1, box=(0, 0, 95, 95))], 2, 100, 100, False)
    assert set(plan) == {0, 1}
    assert plan[0][0][2:] == [.04, .04]
    assert plan[1][0][2:] == [.95, .95]


def test_partially_out_of_frame_boxes_clipped_but_invalid_not_negatives():
    plan, excluded = frame_plan([row(0, box=(90, 0, 120, 100)), row(1, box=(150, 0, 160, 100))], 2, 100, 100, True)
    assert plan == {0: [[.95, .5, .1, 1.]]}
    assert excluded["invalid_bbox"] == 1
