from pathlib import Path

import pytest

from scripts.anti_uav.build_approved_gray_rehearsal import checked_path, parse_label, validate_frame_sets
from scripts.anti_uav.build_manual_gray_video_rehearsal import replace_duplicate_class_slots


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


def test_verified_presence_avoids_reopening_labels(tmp_path: Path):
    pos, neg, newpos, newneg = [tmp_path/name for name in ('pos','neg','newpos','newneg')]
    source = [pos]*8+[neg]*8
    flags = {pos:True, neg:False, newpos:True, newneg:False}
    output, stats = replace_duplicate_class_slots(source, {'clip':[newpos]}, {'clip':[newneg]},
                                                 0.25, 7, verified_presence=flags)
    assert set(output) == set(flags)
    assert stats['output_negative_fraction'] == 0.5
    with pytest.raises(ValueError, match='does not cover'):
        replace_duplicate_class_slots(source, {'clip':[newpos]}, {'clip':[newneg]},
                                      0.25, 7, verified_presence={pos:True})
