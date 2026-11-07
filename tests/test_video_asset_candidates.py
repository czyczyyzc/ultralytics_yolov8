import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.anti_uav.prepare_video_asset_candidates import export, frame_indices, digest
from scripts.anti_uav.import_fixedwing_assets import source_provenance


def test_windows_are_zero_based_half_open_and_deduplicated():
    assert frame_indices([[1, 5], [3, 7]], 2) == {1, 3, 5}
    for windows, step in [([[2, 2]], 1), ([[0, 2]], 0), ([[-1, 2]], 1), ([], 1)]:
        with pytest.raises(ValueError):
            frame_indices(windows, step)


def test_sequential_exports_preserve_pixels_and_provenance(tmp_path):
    video = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (32, 24))
    if not writer.isOpened():
        pytest.skip('MJPG unavailable')
    for value in [10, 60, 110, 160]:
        writer.write(np.full((24, 32, 3), value, np.uint8))
    writer.release()
    plan = dict(videos=[dict(file='source.avi', name='aircraft', windows=[[1, 4]],
                             step=2, roi_xyxy=[4, 6, 28, 22])])
    output = tmp_path / 'out'
    export(tmp_path, plan, output)
    rows = json.loads((output / 'provenance.json').read_text())['records']
    assert [r['source_frame_zero_based'] for r in rows] == [1, 3]
    assert all(r['source_video_sha256'] == digest(video) for r in rows)
    capture = cv2.VideoCapture(str(video))
    for index in range(4):
        ok, frame = capture.read()
        assert ok
        if index in {1, 3}:
            path = output / 'inputs/aircraft' / f'frame_{index:06d}.png'
            assert np.array_equal(cv2.imread(str(path)), frame[6:22, 4:28])
    capture.release()
    inventory = [{k: r[k] for k in ['relative', 'sha256']} for r in rows]
    assert len(source_provenance(output / 'provenance.json', inventory)) == 2
    inventory[0]['sha256'] = 'changed'
    with pytest.raises(ValueError, match='checksum'):
        source_provenance(output / 'provenance.json', inventory)
    with pytest.raises(FileExistsError):
        export(tmp_path, plan, output)


def test_source_paths_cannot_escape_video_root(tmp_path):
    plan = dict(videos=[dict(file='../source.avi', name='aircraft', windows=[[0, 1]],
                            roi_xyxy=[0, 0, 2, 2])])
    with pytest.raises(ValueError, match='escapes'):
        export(tmp_path, plan, tmp_path / 'out')
