from scripts.anti_uav.prepare_snapshot_replacement_cache import pixel_plan


def annotation(frame, status, box=(10, 20, 40, 50)):
    return dict(frame_index=frame, status=status,
                **dict(zip(('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'), box)))


def item(status='submitted'):
    return dict(frame_count=5, frame_width=100, frame_height=100, status=status,
                annotations=[annotation(0, 'confirmed'), annotation(1, 'absent'),
                             annotation(2, 'uncertain'), annotation(3, 'confirmed', (0, 0, 100, 100))])


def test_pending_temporal_donors_do_not_include_unreviewed_absent():
    plan, excluded = pixel_plan(item())
    assert set(plan) == {0, 3}
    assert plan[0] == [[10., 20., 30., 30.]]
    assert excluded['uncertain'] == 1


def test_approved_negative_can_be_a_donor_but_is_not_a_positive_job():
    plan, _ = pixel_plan(item('approved'))
    assert plan[1] == []
    assert [f for f, boxes in plan.items() if boxes] == [0, 3]


def test_full_frame_box_is_not_removed_from_the_training_snapshot():
    plan, _ = pixel_plan(item())
    assert plan[3] == [[0., 0., 100., 100.]]


def test_multi_target_frame_keeps_all_boxes():
    a = item()
    a['annotations'].append(annotation(0, 'confirmed', (60, 60, 80, 80)))
    plan, _ = pixel_plan(a)
    assert len(plan[0]) == 2
