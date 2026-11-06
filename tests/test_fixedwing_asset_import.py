import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.anti_uav.import_fixedwing_assets import alpha_eligible, inventory, publish
from scripts.anti_uav.matte_drone_catalog import digest


def image(path,opaque=False):
    a=np.zeros((16,16,4),np.uint8);a[4:12,4:12]=[50,60,70,255]
    if opaque:a[:,:,3]=255
    Image.fromarray(a).save(path)


def test_opaque_rgba_is_not_transparent(tmp_path):
    folder=tmp_path/'model';folder.mkdir();image(folder/'opaque.png',True);image(folder/'real.png')
    rows=inventory(tmp_path)
    assert [r['native_transparency'] for r in rows]==[False,True]
    with pytest.raises(ValueError,match='opaque'):alpha_eligible(folder/'opaque.png')


def test_publish_preserves_old_and_requires_review(tmp_path):
    old=tmp_path/'old';old.mkdir();image(old/'a.png')
    base=old/'catalog.json';base.write_text(json.dumps(dict(records=[dict(id='1',model='old',cutout='a.png',cutout_sha256=digest(old/'a.png'))])))
    before=base.read_bytes();cand=tmp_path/'candidates';cand.mkdir();image(cand/'b.png')
    b=np.array(Image.open(cand/'b.png'));b[5,5,0]=75;Image.fromarray(b).save(cand/'b.png')
    (cand/'run_config.json').write_text(json.dumps(dict(base_catalog_sha256=digest(base))))
    row=dict(input_id='1',id='335',model='new',cutout='b.png',cutout_sha256=digest(cand/'b.png'))
    (cand/'catalog.json').write_text(json.dumps(dict(complete=True,records=[row])))
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(accepted_input_ids=['1'],notes={})))
    out=tmp_path/'merged';publish(SimpleNamespace(base_catalog=base,candidates=cand,review=review,output=out))
    result=json.loads((out/'summary.json').read_text());assert result['merged_entries']==2
    assert base.read_bytes()==before and result['existing_training_configuration_changed'] is False
    review.write_text(json.dumps(dict(accepted_input_ids=['99'],notes={})))
    with pytest.raises(ValueError,match='Unsegmented'):
        publish(SimpleNamespace(base_catalog=base,candidates=cand,review=review,output=tmp_path/'bad'))
