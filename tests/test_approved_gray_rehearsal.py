from pathlib import Path

import pytest

from scripts.anti_uav.build_approved_gray_rehearsal import checked_path, parse_label, validate_frame_sets


def test_uncertain_frames_are_not_negatives():
    m = dict(video=dict(frameCount=4), frames=dict(indexBase=0, includedFrameIndices=[0,2,3],
             negativeFrameIndices=[2], excludedUncertainFrameIndices=[1], excludedUnreviewedFrameIndices=[]))
    assert validate_frame_sets(m) == ({0,2,3}, {2})
    m['frames']['negativeFrameIndices'].append(1)
    with pytest.raises(ValueError):
        validate_frame_sets(m)


def test_frame_classification_must_be_complete():
    with pytest.raises(ValueError):
        validate_frame_sets(dict(video=dict(frameCount=3), frames=dict(indexBase=0,
                            includedFrameIndices=[0], negativeFrameIndices=[])))


@pytest.mark.parametrize('label', ['0 nan .5 .1 .1', '1 .5 .5 .1 .1', '0 .5 .5 0 .1', '0 .99 .5 .1 .1'])
def test_invalid_boxes_rejected(label):
    with pytest.raises(ValueError):
        parse_label(label)


def test_empty_and_valid_multibox_labels():
    assert parse_label('') == []
    assert len(parse_label('0 .5 .5 .1 .1\n0 .3 .3 .1 .1\n')) == 2


def test_manifest_path_cannot_escape(tmp_path: Path):
    with pytest.raises(ValueError):
        checked_path(tmp_path, '../outside.txt')
    assert checked_path(tmp_path, 'labels/1.txt') == tmp_path/'labels/1.txt'
