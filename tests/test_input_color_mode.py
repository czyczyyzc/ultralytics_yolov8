import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts.anti_uav import label_pool_sampling

spec=importlib.util.spec_from_file_location('input_color',Path(label_pool_sampling.__file__).resolve().parents[2]/'ultralytics/data/color.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gray3_preserves_source_pixels_geometry_and_labels():
    image=np.random.default_rng(13).integers(0,256,(29,53,3),dtype=np.uint8)
    original=image.copy();boxes=np.array([[.2,.3,.1,.2]])
    labels=dict(img=image,bboxes=boxes,im_file='original.png')
    result=module.input_color_transform('gray3')(labels)
    expected=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    assert result['img'].shape==image.shape
    assert all(np.array_equal(result['img'][:,:,c],expected) for c in range(3))
    assert np.array_equal(image,original)
    assert result['bboxes'] is boxes
    assert result['im_file']==labels['im_file']


def test_gray3_is_idempotent_and_unknown_mode_rejected():
    image=np.repeat(np.arange(20,dtype=np.uint8).reshape(4,5,1),3,axis=2)
    assert np.array_equal(module.GrayscaleThreeChannel()(dict(img=image))['img'],image)
    assert module.input_color_transform('source') is None
    with pytest.raises(ValueError,match='Unknown input_color'):
        module.input_color_transform('grey')


@pytest.mark.parametrize('image',[np.zeros((2,3),np.uint8),np.zeros((2,3,3),np.float32)])
def test_gray3_rejects_wrong_shape_or_dtype(image):
    with pytest.raises(ValueError,match='uint8 HWC BGR'):
        module.GrayscaleThreeChannel()(dict(img=image))
