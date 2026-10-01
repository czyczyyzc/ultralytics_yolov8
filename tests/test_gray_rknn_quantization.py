import numpy as np
import pytest
import torch

from scripts.anti_uav.compare_gray_rknn_quantization import raw_prediction


@pytest.mark.parametrize("branches", [3, 4])
def test_raw_prediction_dfl_uniform_and_confidence(branches):
    outputs = []
    for _ in range(branches):
        outputs.extend([np.zeros((1, 64, 1, 2), np.float32),
                        np.full((1, 1, 1, 2), .25, np.float32),
                        np.full((1, 1, 1, 2), .25, np.float32)])
    result = raw_prediction(outputs, (32, 64))
    assert result.shape == (1, 5, branches * 2)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result[0, :4, :2], torch.tensor(
        [[16., 48.], [16., 16.], [480., 480.], [480., 480.]]))
    assert torch.all(result[:, 4] == .25)


def test_raw_prediction_rejects_wrong_layout_or_integer_outputs():
    with pytest.raises(ValueError, match="9 or 12"):
        raw_prediction([np.zeros((1, 64, 1, 1))])
    outputs = [np.zeros((1, 64, 1, 1), np.int8), np.ones((1, 1, 1, 1), np.float32),
               np.ones((1, 1, 1, 1), np.float32)] * 3
    with pytest.raises(ValueError, match="dequantized"):
        raw_prediction(outputs)
    outputs[0] = np.zeros((1, 1, 1, 64), np.float32)
    with pytest.raises(ValueError, match="NCHW"):
        raw_prediction(outputs)
