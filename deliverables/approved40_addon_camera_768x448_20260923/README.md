# RK3588S live-camera latency: Frozen-P3 + Add-on P2

Test date: 2026-09-23. Board: Orange Pi CM5 / LubanCat-4, RK3588S,
Linux 5.10.198. This is the 40-approved-video + 328-cutout-asset
Frozen-P3/Add-on-P2 checkpoint, not the older 28-video release.

## Input and model

- Source checkpoint on server:
  `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922/training_addon/p2/weights/best.pt`
- Toolkit2 2.3.2, RK3588 target, INT8, 384 train-only calibration images.
  Video00004 and Video00009 were excluded from calibration.
- Camera: `/dev/video11`, 1920x1080 BA81 raw8 treated as monochrome,
  120 FPS; copied to RGB for the model, without demosaicing.
- Requested 768x432 content is letterboxed with 8 pixels of value 114 on
  each vertical edge. The **actual RKNN tensor is 768x448**, because YOLO
  export requires a stride-32-aligned input. Four detector scales produce
  12 RKNN outputs. Do not feed a 768x432 tensor to this model.
- Local RKNN: `detector_768x448_int8.rknn`, SHA256
  `b3c5a3b1ba9f655115ec014e776129f2ce998f1c1499f2e0515bfcbe030566be`.
- Server export directory:
  `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_addon_camera_768x448_20260923/`.
- Board deployment:
  `/home/orangepi/deployments/approved40_addon_768x448_20260923/`.

## Measured live-camera performance

The native C++ path is V4L2 read -> gray/RGB preparation -> three independent
RKNN contexts (NPU cores 0/1/2) -> GMC + Dist tracking -> result. GMC runs
concurrently with detector inference. Each steady run excludes 100 warmup
frames. `latest` capture policy uses four camera buffers and three in-flight
frames. Confidence is 0.03, NMS IoU is 0.45. No Python, display, or encoding
is in the timed path.

| Run | Measured frames | Result FPS | `read()` start to result mean / P95 | V4L2 driver timestamp to result mean / P95 | Source frames skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three NPU cores, long | 12,000 | **108.31** | 27.58 / 32.81 ms | **36.61 / 42.78 ms** | 9.7% |
| Three NPU cores, short | 2,000 | 108.87 | 27.43 / 32.54 ms | 36.47 / 42.50 ms | 9.4% |
| One NPU core, short | 2,000 | 40.25 | 24.80 / 27.27 ms | 36.22 / 40.48 ms | 66.5% |

The long run's average per-frame stage times were: dequeue/read 1.44 ms,
preprocessing 1.91 ms, RKNN 23.61 ms, postprocessing 0.21 ms, GMC 2.90 ms,
and Dist association 0.026 ms. These stages overlap, so their means must
not be added to infer pipeline latency. Driver timestamp to dequeue alone
averaged 9.97 ms. The three workers processed 4000, 3938, and 4162 frames.
The board ended near 47-48 C, with NPU 1.0 GHz and DDR 1.56 GHz.

Three independent fresh-process first-frame trials (`--warmup 0`) gave
medians of **189.11 ms** from process entry, **134.03 ms** from STREAMON /
first read, and **33.24 ms** from the first driver's frame timestamp to
complete result. The initial camera STREAMON call itself takes about 100 ms.
These are different start events, not conflicting measures of the same delay.

## Interpretation and limits

- Three-core throughput is about 2.69 times one-core throughput. The
  driver-timestamp-to-result mean remains nearly unchanged (36.61 versus
  36.22 ms), as expected for parallel contexts, but read-start-to-result
  rises by 2.78 ms because of scheduling and ordered output.
- At 120 FPS input, the 108.31 FPS `latest` result is not a result for every
  sensor frame. The 12,000-result run skipped about 9.7% of source frames.
  Use a lower camera FPS or faster inference if every frame is required.
- The `--preprocess fused` request used `opencv_fallback` in this test:
  1920x1080 -> 768x432 is 2.5x, while the existing fused kernel supports
  only the 2x path. Thus this is not a RGA/zero-copy benchmark.
- Driver MONOTONIC timestamps are not verified sensor exposure times; the
  measurements exclude image display, video encode, and network transport.
- The indoor camera scene has no annotated UAV trajectory. INT8 accuracy,
  small-target recall, and tracking ID quality at this new input size were
  **not** assessed. Performance alone does not establish deployment quality.
- The short three-core run was produced before a metadata fix and its
  `input_wh` field incorrectly says 960x544. It used the same 768x448 RKNN
  model; the 12,000-frame and one-core runs query and report the correct
  tensor dimensions. Prefer the 12,000-frame run for citation.
- The older 960x544 camera benchmark used a different 28-video checkpoint,
  so its FPS and latency are not a controlled resolution-only comparison.

## Evidence and reproduction

`latest3_12000/summary.json` and `latency.csv` are the primary measurements.
`latest1_2000/` is the one-core control. `startup_w0_{1,2,3}/` are separate
cold-process first-frame trials. `comparison.json` is independently checked
against every CSV row by:

```sh
python scripts/anti_uav/summarize_camera_benchmark.py \
  deliverables/approved40_addon_camera_768x448_20260923
```

On the board, the C++ executable and libraries are in the deployment's `bin/`
directory; the Git checkout is `/home/orangepi/ultralytics_yolov8_ziye_code`.
Use a fresh output directory and the deployed model with the camera launcher:

```sh
cd /home/orangepi/ultralytics_yolov8_ziye_code
ANTI_UAV_CAMERA_DEPLOY=/home/orangepi/deployments/approved40_addon_768x448_20260923 \
  bash scripts/anti_uav/dist_native/run_camera_on_board.sh \
  --model /home/orangepi/deployments/approved40_addon_768x448_20260923/detector_768x448_int8.rknn \
  --frames 12000 \
  --output /home/orangepi/deployments/approved40_addon_768x448_20260923/retest
```

The launcher has a legacy model default, hence the explicit `--model` override.
For one core, append `--workers 1 --inflight 1`. For a first-frame trial,
append `--frames 1 --warmup 0` and use another fresh output directory.
