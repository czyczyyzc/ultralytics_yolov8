import numpy as np
import pytest
import torch

from scripts.anti_uav.compare_gray_rknn_quantization import raw_prediction
from scripts.anti_uav.export_detector_rkopt_onnx import prepare_signed_p3_export, validate_rkopt_shapes
from ultralytics.nn.modules.head import Detect


def test_folded_export_matches_native_signed_dfl_projection():
    head = Detect(1, (16,32,64)).eval()
    head.stride = torch.tensor([8.,16.,32.])
    head.enable_p3_signed_dfl()
    with torch.no_grad():
        head.p3_negative_bin.bias.fill_(.5)
        features = [torch.rand(1,c,s,s) for c,s in ((16,8),(32,4),(64,2))]
        reference = head(features.copy())[0]
        assert prepare_signed_p3_export(head)
        head.export, head.format = True, "rknn"
        outputs = [x.numpy() for x in head(features.copy())]
    validation = validate_rkopt_shapes([list(x.shape) for x in outputs])
    assert validation["branches"][0]["dfl_bins"] == list(range(-1,16))
    assert validation["branches"][1]["dfl_bins"] == list(range(16))
    torch.testing.assert_close(raw_prediction(outputs,(64,64)),reference,rtol=1e-5,atol=1e-4)


def test_signed_decoder_recovers_an_off_grid_box_and_rejects_wrong_grid():
    bbox = np.full((1,68,8,8), -30, np.float32)
    for side,dist in enumerate((-.125,-.125,.5,.375)):
        lo = int(np.floor(dist))
        fraction = dist-lo
        bbox[0,side*17+lo+1,0,0] = np.log(1-fraction)
        bbox[0,side*17+lo+2,0,0] = np.log(fraction)
    cls = np.ones((1,1,8,8),np.float32)
    outputs = [bbox,cls,cls]+[np.zeros((1,64,2,2),np.float32),cls[:,:,:2,:2],cls[:,:,:2,:2]]*2
    assert raw_prediction(outputs,(64,64))[0,:4,0].tolist() == pytest.approx([6.5,6.,3.,2.])
    with pytest.raises(ValueError,match="stride-8"):
        raw_prediction(outputs,(64,128))
