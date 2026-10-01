import numpy as np
import pytest
import torch
import json

from scripts.anti_uav import compare_gray_rknn_quantization as comparison
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


def test_merge_restores_frame_order_and_rejects_missing_shards(tmp_path, monkeypatch):
    # The production reference uses NumPy 1.x; keep this test valid on newer NumPy too.
    if not hasattr(np, "trapz"):
        monkeypatch.setattr(np, "trapz", np.trapezoid, raising=False)
    monkeypatch.setattr(comparison, "EXPECTED", {s: (2, 1) for s in comparison.SPLITS})
    source = tmp_path / "source.txt"
    source.write_text("/frame0.png\n/frame1.png\n")
    shards = []
    for index in range(2):
        path = tmp_path / str(index)
        path.mkdir()
        shards.append(path)
        # Put negative frame1 in shard0 so naive concatenation has the wrong order.
        frame = 1 - index
        protocol = dict(artifacts={}, target="rk3576", toolkit_version="2.3.2", input_hw=[544, 960],
            padding=114, thresholds=[.01, .03, .05], shards=2, shard_index=index, smoke_limit_per_split=0,
            rebuilt_matches_delivered=False, rebuilt_sha256=str(index), datasets={})
        backends = {b: {} for b in comparison.BACKENDS}
        for split in comparison.SPLITS:
            protocol["datasets"][split] = dict(source_list=str(source))
            (path / f"{split}.txt").write_text(f"/frame{frame}.png\n")
            metrics = {}
            for conf in (.01, .03, .05):
                q = f"native/c{conf:.2f}/"
                metrics.update({q+"TP": int(frame == 0), q+"FP": int(frame == 1), q+"FN": 0,
                    q+"FRAMES": 1, q+"long_4to8px/GT": int(frame == 0),
                    q+"long_4to8px/R": 1. if frame == 0 else None})
            for backend in comparison.BACKENDS:
                backends[backend][split] = metrics
                np.savez_compressed(path / f"{split}_{backend}_pr.npz",
                    tp=np.full((1, 10), frame == 0), conf=np.array([.5]), pred_cls=np.zeros(1),
                    target_cls=np.zeros(int(frame == 0)), prediction_frame_index=np.array([frame]),
                    target_frame_index=np.full(int(frame == 0), frame))
        (path / "results.json").write_text(json.dumps(dict(protocol=protocol, backends=backends,
            raw_export_max_error=dict(box_xywh=0., class_score=0.))))
        (path / "status.json").write_text(json.dumps(dict(stage="complete")))
    with pytest.raises(ValueError, match="Missing or duplicate"):
        comparison.merge_shards(tmp_path / "incomplete", shards[:1])
    output = tmp_path / "merged"
    comparison.merge_shards(output, shards)
    result = json.loads((output / "results.json").read_text())
    pooled = result["backends"]["pt_fp32"]["pooled_native"]
    assert pooled["native/c0.03/TP"] == 2
    assert pooled["native/c0.03/FP"] == 2
    assert pooled["native/c0.03/R"] == 1
    with np.load(output / "Video00004_test_pt_fp32_pr.npz") as arrays:
        assert arrays["tp"][:, 0].tolist() == [True, False]
