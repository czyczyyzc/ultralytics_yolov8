# Pure P3 Camera Latency: Removing Add-on P2 Computation

Deployment handover (Chinese): [DEPLOYMENT_HANDOVER_ZH.md](DEPLOYMENT_HANDOVER_ZH.md).

Measured 2026-09-21 on the live RK3588S CM5 camera. The complete Add-on P2 branch
is absent from this model, not merely ignored during postprocessing. No model
retraining was performed. The source is the P3 checkpoint from the same expanded
28-video training run, before frozen-P3 add-on training.

## Matched Long Runs

Both models: 960x544 INT8, conf 0.03, NMS IoU 0.45, native C++ detector + Dist +
GMC, three independent NPU contexts (masks 0/1/2), three in-flight slots, latest
capture with four V4L2 buffers, overlapped sensor/model startup, no explicit NPU
warmup. Each test processed 12,000 live-camera outputs, excluding the first 100
outputs from steady statistics. Same binary, shared libraries, board, kernel,
camera format and frequency policy; sequential runs, not identical sensor frames.

| Measurement | Frozen-P3 + Add-on P2 | Pure P3-P5 |
| --- | ---: | ---: |
| Full pipeline output FPS | 86.58 | 120.00 |
| NPU inference per frame, mean | 31.56 ms | 21.20 ms |
| Camera read call to result, mean | 34.53 ms | 24.94 ms |
| Driver frame timestamp to result, mean | 43.96 ms | 29.91 ms |
| Driver frame timestamp to result, P95 | 49.59 ms | 31.84 ms |
| Driver sequence skipping | 27.8% | 0.0% |
| Peak sampled temperature | 45.31 C | 46.23 C |

Pure P3 reduced mean NPU time by 32.8% and driver-to-result latency by 32.0%.
Throughput increased 38.6%, reaching the camera's approximately 120-FPS supply.
This is not a measurement of P3's unconstrained/offline maximum throughput.
120 FPS means results arrive about every 8.33 ms; each processed frame still
takes about 29.91 ms from its driver timestamp to result.

A subsequent 2,000-output P3+P2/P3 recheck produced 86.14/119.99 FPS and
44.05/29.95 ms driver-to-result mean, respectively. P3 again had zero source
sequence gaps. A 6,000-output P3 `fresh`-policy run was worse than latest:
112.23 FPS, 30.01 ms mean, 32.48 ms P95, 6.7% skipped. Keep the latest policy;
the earlier fresh-policy 1,000-output result alone was not sufficient evidence.

The P3 run lasted approximately 100 seconds with zero sequence gaps and zero
explicitly drained buffers. This is not an indefinite thermal guarantee.
GMC averaged 3.20 ms and overlaps detector inference. Association averaged
0.0062 ms, but the indoor scene had only five detections and one displayed-track
output; this does not establish busy-scene association latency or tracking quality.

For comparison, one single-core P3 worker (1,000 outputs, same latest policy)
ran at 49.03 FPS: NPU 19.35 ms, read-to-result 20.33 ms, driver-to-result
35.53 ms (P95 42.90 ms), skipping 59.1% of source sequence positions. Lower
single-frame inference time did not imply lower live frame age because capture
waiting and buffering differ.

## Startup and Meaning of Latency

Five new-process trials with overlapped startup, medians:

| Measurement | P3 + P2 | Pure P3 |
| --- | ---: | ---: |
| Program entry to first result | 141.37 ms | 133.47 ms |
| STREAMON invocation to first result | 137.93 ms | 130.00 ms |
| First read invocation to first result | 36.20 ms | 26.73 ms |

The sensor STREAMON call still consumes roughly 102-104 ms, largely overlapping
model loading. Model removal has a much smaller impact on total startup than on
steady inference. These tests do not include shell/dynamic-loader time or board
power-on. Driver MONOTONIC timestamps (flags 8193) have not been independently
verified against exposure; no display/encoding or photon-to-result timing is claimed.

## Model and Launch

Board model:
`/home/orangepi/deployments/expanded28_p3_camera_20260921/detector_960x544_int8.rknn`

Server export:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/expanded28_p3_camera_20260921/detector_p3_960x544_int8.rknn`

Local model: `detector_p3_960x544_int8.rknn` alongside this report. Full provenance
and hashes are in `model_manifest.json`. The P3 deployment reuses the tested
latency deployment's `bin` and the original camera deployment's `deps` through
symlinks, so retain those directories. It does not overwrite the P3+P2 model.

```sh
cd /home/orangepi/ultralytics_yolov8_ziye_code
bash scripts/anti_uav/dist_native/run_camera_p3_on_board.sh \
  --frames 12000 \
  --output /home/orangepi/deployments/expanded28_p3_camera_20260921/new_run
```

The export graph has nine outputs (box/class/score sum for stride 8/16/32), not
twelve outputs. No stride-4 feature map or Add-on P2 adapter is executed.
`p3_equivalence.json` checks the FP32 P3/P4/P5 raw outputs against the frozen
branches of the deployed add-on checkpoint on one seeded 256x256 input; all
maximum absolute differences are zero. INT8 quantization is separately compiled
and is not claimed to be bit-identical to those FP32 outputs.

Export used the existing `export_detector_rkopt_onnx.py` with `--imgsz 544,960`,
then `build_rknn.py --target rk3588 --quantize`, toolkit 2.3.2 and the same 384
calibration images as P3+P2. No Video00004/Video00009 images entered calibration.
The compiler emitted an outlier-weight warning, retained in `build_rknn.log`.

## Evidence and Accuracy Scope

`comparison.json` is reproduced by running the offline
`scripts/anti_uav/summarize_camera_benchmark.py` on this directory. It checks all
CSV rows for monotonic timestamp ordering, model hash consistency, sequence gaps
and summary agreement. Raw CSV/JSON and first camera images are retained here
and on the board. The P3+P2 control is
`../expanded28_camera_latency_20260921/overlap3_12000/`.

This experiment establishes latency/throughput, not detection/track accuracy.
The real camera views an upside-down, partially obscured indoor scene without
an annotated UAV trajectory. Removing P2 can change small-target recall; held-out
video evaluation is required before adopting pure P3 as the accuracy default.
The original P3+P2 deployment remains available and unchanged.
