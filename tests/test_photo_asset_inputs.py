import json

from PIL import Image
import pytest

from scripts.anti_uav.stage_photo_asset_inputs import digest, stage


def test_photo_staging_preserves_bytes_and_surface_metadata(tmp_path):
    photo = tmp_path / 'belly.png'
    Image.new('RGB', (16, 12), 'silver').save(photo)
    plan = dict(photos=[dict(source=str(photo), visible_surface='underside')])
    output = tmp_path / 'out'
    stage(plan, output)
    rows = json.loads((output / 'provenance.json').read_text())['records']
    assert len(rows) == 1
    assert rows[0]['visible_surface'] == 'underside'
    assert rows[0]['source_kind'] == 'supplied_photo'
    assert rows[0]['sha256'] == digest(photo)
    assert (output / 'inputs/beluga/01.png').read_bytes() == photo.read_bytes()
    with pytest.raises(FileExistsError):
        stage(plan, output)
