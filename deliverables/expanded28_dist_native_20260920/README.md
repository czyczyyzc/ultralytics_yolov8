# Expanded-28 Native C++ RKNN + Dist + GMC / RK3588S

## Deployment

Board: Orange Pi CM5, RK3588S, Ubuntu 22.04, kernel 6.1.99,
RKNN runtime 2.3.2. Production inference is **entirely native C++**. The executable
does not start or load Python. `native_runtime_audit.txt` records the running
process and its dependencies. Python is used only by separate offline validation
and measurement tools, never by the inference executable.

Board executable and new libraries:
`/home/orangepi/deployments/expanded28_dist_native_20260920/`.
Source: `/home/orangepi/ultralytics_yolov8_ziye/scripts/anti_uav/dist_native/`.
Executable source commit: `7d6e969`; verification scripts: `ad04d94`.

```bash
cd /home/orangepi/ultralytics_yolov8_ziye
bash scripts/anti_uav/dist_native/run_on_board.sh \
  --output /home/orangepi/deployments/expanded28_dist_native_20260920/my_run \
  --save-observations
```

The output directory must not already exist. Add `--frames 2000` for a short
test. Omit `--save-observations` if per-frame JSON output is not needed.
See `scripts/anti_uav/dist_native/README.md` for building and explicit path
overrides. This is an optimization overlay, not a standalone model/environment
archive. The existing detector library, video and model remain dependencies:

- Model: `/home/orangepi/deployments/expanded28_dist_20260920/detector_960x544_int8.rknn`
- Detector library: `/home/orangepi/deployments/expanded28_dist_opt_20260920/libanti_uav_detector.so`
- Video: `/home/orangepi/deployments/expanded28_dist_20260920/Video00009_original.mp4`

The detector remains the 28-video Frozen-P3 + Add-on P2, **960x544 INT8**, unchanged
from the previous deployment. RKNN SHA256:
`2afa76f9f093265b0e347d26285ed15c98c449d958eed040f27e7345f29deb53`.

## Runtime Changes

The native executable overlaps CPU OpenCV decoding, three independent RKNN
workers, and one ordered GMC worker. Frames go to ready workers with NPU masks
0/1/2; a bounded queue permits at most nine in-flight frames. Native Dist consumes
results in source frame order. No frame skipping, cached detection inference,
Python bridge, ReID, predicted-only display boxes or GT is used at runtime.
The decoder remains CPU OpenCV, not hardware MPP/RGA decoding.

Native Dist preserves the executed public implementation's Kalman state,
cost-limited LAPJV matching, confirmation/lost lifecycle, duplicate removal and
current-detection output behavior. It is not the old RK-BoT-SORT renamed as Dist.
The public adaptation does not claim absent paper-specific FLIT/L2-IoU features.

GMC resizes first to 320x180, uses at most 128 corners, refreshes features every
five frames or when needed, and processes optical flow on every frame. Native
buffers and LK pyramids are reused. Detector thresholds remain conf 0.03 and NMS
IoU 0.45; tracker high/low/new/match remain 0.03/0.01/0.10/0.8. No area cutoff is
introduced. All displayed boxes come from their frame's actual detector output.

## Verified Results

All pipeline rates include real video decoding, preprocessing, NPU inference,
postprocessing, live GMC and association; they exclude display/video encoding.
The first 100 frames are warmup. Per-frame JSON recording is enabled.

| Pipeline | Frames | FPS | Last 2,201 frames FPS |
| --- | ---: | ---: | ---: |
| Original Python/public GMC baseline | 14,201 | 18.36 | 16.28 |
| Previous optimized Python/compact GMC | 14,201 | 57.59 | 53.73 |
| Native C++ short run | 2,000 | 85.40 | - |
| Native C++ complete video | 14,201 | 79.57 | 76.87 |
| Native C++ warm-board repeat | 14,201 | 79.65 | 77.29 |

The first native full run has mean association **0.0480 ms** versus **2.2732 ms**
previously. Mean GMC is **4.8262 ms** versus **8.1502 ms**. Combined GMC and
association is **4.8742 ms**, not 0.048 ms. The association stage is now in the
same order of magnitude as historical RK-BoT-SORT association (~0.01-0.03 ms),
but those historical workloads are not identical and did not have this live GMC.

Mean read-to-ordered-result latency is **49.53 ms**, p95 **64.10 ms**, versus
previous **70.07 / 94.26 ms**. Three workers processed 4,753 / 4,730 / 4,718 frames.
These overlapping stage times cannot be summed to infer pipeline FPS.

Thermal conditions differ: the previous optimized run started around 67.5 C and
reached 84-85 C with NPU throttling. This first native full run started around
36 C and finished around 73-75 C, with sampled NPU frequency at 1 GHz. Therefore
the observed 38.2% FPS increase is not a controlled estimate of code-only gain.
No thermal protections, frequency limits or services were changed for this test.

The warm-board repeat started at 62.8 C and ended at 73.9 C (zone 0); the highest
sample across all sensors was 80.4 C. NPU samples remained at 1 GHz. It matched
the previous pipeline exactly on all frames too (`hot_equivalence.json`). Its
mean association/GMC/ordered latency were **0.0464 / 4.8445 / 49.38 ms**. Two
complete runs represent about six minutes of processing, not a guarantee for
indefinite operation under different cooling conditions.

Three fresh executable launches gave **254.37 / 239.97 / 242.43 ms** to the first
result: median **242.43 ms**, versus previous **713.38 ms**. After initialization,
first-frame read-to-result median was **87.06 ms**, versus previous **105.35 ms**.
`startup_native/summary.json` stores all trials. An external Python stopwatch
launched the native executable for this offline test; Python is not inside the
inference process. OS/filesystem caches were not cleared. This is neither
board-power-on nor camera-to-display latency.

### Correctness And Quality

All **14,201** frames and **8,169** displayed observations match the previous
optimized pipeline exactly: detection boxes/scores, IDs/current detector indices,
and GMC affine matrices. `full_equivalence.json` records zero differing frames.
The synthetic tracker suite covers 690 frames including threshold boundaries,
duplicates, crossings and expiry/reappearance. Maximum tested Kalman numerical
difference is 1.3e-9; IDs and output boxes are exact. GMC raw/cached native tests
also matched the Python implementation on 2,000 decoded frames.

Reviewed Video00009 quality at IoU 0.5 is unchanged:

| Output | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Detector | 5,317 | 3,539 | 2,665 | 60.04% | 66.61% |
| Dist + compact GMC, old and native | 5,283 | 2,885 | 2,699 | 64.68% | 66.19% |

Continuous-GT identity changes remain 9; adjacent-TP-frame identity changes
remain 1. These diagnostics are not standard MOT IDSW. One observation is on an
excluded uncertain frame, so scored tracker outputs total 8,168. This is a
performance-preserving implementation optimization, not a new accuracy gain or
independent generalization evaluation. Unconfirmed detector boxes can still be
suppressed by the unchanged tracker rules.

`tracker_equivalence.json` includes a ctypes replay microbenchmark for migration
validation only. Its 0.0443 ms value is not native pipeline FPS and does not
include live GMC. The actual production association timing is in the full-run
`summary.json` above.

Additional checks: eight strict-comparison unit tests, four GMC unit tests and
four native CLI error-path checks passed. Missing arguments, unknown options and
existing output directories fail instead of silently overwriting results.

## Result Locations

Local results: `/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/expanded28_dist_native_20260920/`.
Server 47 results: `/mnt/chenziye/codes/ultralytics_yolov8/deliverables/expanded28_dist_native_20260920/`.
The server-side scoring run is also retained under
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/expanded28_dist_native_20260920/quality_cpp_full14201/`.
Code was committed/pushed locally and pulled into both server and board repos.
Existing model and older deployment directories were not replaced.
