# Expanded-28 RKNN + Dist + Efficient GMC / RK3588S

## Deployment

Board deployment: `/home/orangepi/deployments/expanded28_dist_opt_20260920`.
Board repository: `/home/orangepi/ultralytics_yolov8_ziye`.
The original deployment is retained at
`/home/orangepi/deployments/expanded28_dist_20260920`.
The optimized launcher reuses its virtual environment, RKNN model, upstream
Dist source and Video00009. This directory is an optimization overlay, not a
standalone environment/model archive. `ANTI_UAV_BASE` and `ANTI_UAV_REPO` may
override these paths.

Model remains the 28-video Frozen-P3 + Add-on P2, **960 x 544 INT8**, with
RKNN runtime 2.3.2. No retraining, input reduction or threshold increase was used.

Checkpoint on server 47:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_addon/p2/weights/best.pt
```

RKNN on the board:

```text
/home/orangepi/deployments/expanded28_dist_20260920/detector_960x544_int8.rknn
```

RKNN SHA256: `2afa76f9f093265b0e347d26285ed15c98c449d958eed040f27e7345f29deb53`.
Optimized native library SHA256:
`f48c561fb661d5ed1d8e42a6cd067755d082e6868e0f90e8a1de68da03d6f6f0`.
The final runtime/launcher was measured at repository commit `1080c67`.
See the original deployment's `model_manifest.json` for calibration and export.
Board inference is native C++ RKNN; no PyTorch is required on the board.
Public Dist association remains NumPy/SciPy/lap; GMC is CPU OpenCV.
This is the public repository BOTSORT adaptation, not a claim of implementing
the paper's absent FLIT/L2-IoU features.

## Pipeline And Correctness

The bounded pipeline overlaps decoding, three independent per-frame RKNN workers
and one ordered live GMC worker. Each NPU context has separate I/O buffers and is
bound to core 0, 1 or 2; duplicated contexts share model weights. Association
consumes detections and camera motion in original frame order. At most nine frames
are in flight; no video frames are skipped and no detections/warps are cached for
timing. Video is still decoded using OpenCV on the CPU, not hardware MPP decoding.

Frames are dispatched to an available NPU worker rather than forcing a fixed
round-robin assignment when workers have unequal CPU-side delays. Completion order
does not change the original-frame order used by the tracker. Contexts are never
used concurrently by two frames.

OpenCV and BLAS use one thread. Host work is restricted to big cores 4-7, avoiding
small-core migration and nested thread contention. CPU affinity and NPU core masks
are separate settings. The original three-core benchmark already used all three
NPU cores; its 49.29 FPS was not evidence of a missing NPU worker.

GMC modes:

- `public`: upstream sparse optical flow at half source resolution, preserving
  the original algorithm. First 2,000 frames matched original tracks and IDs exactly
  after the asynchronous scheduling change.
- `compact`: explicit algorithm variant with a bounded image size and feature
  budget, vectorized point filtering, inlier validation and identity fallback.
  The efficient profile uses width 320 (320 x 180 for this video), 128 corners,
  nominal feature refresh every five frames. Optical flow and association still
  run on **every frame**. Features are refreshed early when necessary.
- `--gmc-resize-first`: avoids full-resolution BGR-to-gray conversion by resizing
  first. The full server replay on this grayscale Video00009 produced identical
  displayed observations to compact gray-first. Color inputs may differ due to
  interpolation/rounding and need separate quality validation.

Detector settings remain conf 0.03, NMS IoU 0.45, padding 114, max 100 boxes.
Tracker high/low/new thresholds remain 0.03/0.01/0.10, score fusion is off, ReID is
off, buffer is 100 source frames at 100 FPS. No area filtering, GT-based runtime
decisions or extrapolated display boxes are added. Outputs are confirmed tracks
associated with actual current-frame detections; this can still suppress some
unconfirmed detector boxes.

## Measured Results

All rates below include video decoding, native preprocessing, RKNN and DFL/NMS;
tracking rows also include live GMC and association. Warmup is 100 frames.

| Configuration | Total frames | FPS | Last 2,201 frames FPS |
| --- | ---: | ---: | ---: |
| Original detector baseline | 2,000 | 49.29 | - |
| Optimized detector short run | 2,000 | 86.82 | - |
| Optimized detector full run, round-robin / queue 4 | 14,201 | 64.62 | 60.27 |
| Original detector + public Dist/GMC baseline | 14,201 | 18.36 | 16.28 |
| Public GMC, asynchronous scheduling / queue 4 | 2,000 | 24.84 | - |
| Compact GMC, round-robin / queue 4 | 14,201 | 45.94 | 38.88 |
| **Delivered profile: compact GMC, ready dispatch / queue 9** | **14,201** | **57.59** | **53.73** |

The final complete pipeline is **3.14x** the original complete-pipeline average.
Mean read-to-ordered-result latency decreased from **326.7 ms to 70.1 ms**;
final p95 is **94.3 ms**. Nine is the in-flight upper bound, not a forced wait
for a batch of nine frames. Ready dispatch avoids unnecessary per-worker backlog.

Final average stage times: decode 11.15 ms, preprocessing 5.25 ms, per-worker NPU
40.93 ms, postprocessing 0.43 ms, GMC 8.15 ms, association 2.27 ms. The original
combined tracker/GMC stage averaged 46.73 ms; the final combination is 10.42 ms.
The three NPU workers processed 4,731 / 4,728 / 4,742 frames respectively.

Three fresh-process startup trials gave a median **713.4 ms** launch-to-first
completed result (original 734.5 ms). Median first-frame read-to-result after
initialization was **105.4 ms** (original 126.1 ms). Most of the latency gain is
in sustained processing, not startup. `startup_final/summary.json` preserves all
three trials, including imports and model initialization. These are warm-OS/cache
process starts, not power-on or camera-to-screen latencies.

Final SoC temperature rose from **67.5 C** to around **84-85 C**, with NPU samples
showing 1 GHz and 800 MHz. The 86.82 FPS short run cannot be promised continuously
with the current thermal conditions. Stable near-peak performance needs sustained
cooling and a new extended test; no thermal protection was bypassed.

Alternative probes are retained in `comparison.json`: one all-core-mask context
33.04 FPS, three all-core-mask contexts 38.47 FPS, versus split-core short-run
86.82 FPS. Changing CPU scheduling to a shared affinity set or restricting work
to two big cores did not improve the observed short runs. These probes have
different temperatures and durations; they do not establish isolated causal gains.
The delivered strategy remains three separate NPU core masks, not three instances
all competing with an all-core mask.

### Quality Check

Every detector box/score on all **14,201 frames** in the final run is bit-identical
to the original full RKNN run. No confidence threshold or detector resolution was
changed. Final detector-only Precision/Recall at IoU 0.5 are **60.04% / 66.61%**.

| Actual board tracking outputs | TP | FP | FN | Precision | Recall | Continuous-GT ID changes | Adjacent-TP ID changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original public GMC | 5,293 | 2,868 | 2,689 | 64.86% | 66.31% | 8 | 3 |
| Final compact GMC | 5,283 | 2,885 | 2,699 | 64.68% | 66.19% | 9 | 1 |

The faster GMC costs 10 TP and adds 17 FP here: Recall -0.125 percentage points,
Precision -0.178 percentage points. It is **not an accuracy-identical replacement**
for public GMC and does not improve every identity metric. Use public mode when
preserving the original motion algorithm is more important than throughput.

Changing only compact-GMC preprocessing order and ready-worker scheduling produced
identical full-video warps/tracks to the corresponding compact gray-first run.
The upstream-to-compact algorithm change, not detector inference, causes the
quality difference above. Four GMC unit tests and two pipeline ordering/failure
cleanup tests passed on both server 47 and the board.

Primary evidence: `final_ready9/summary.json`, `final_ready9/observations.jsonl`,
`full_detector_optimized/summary.json`, `startup_final/summary.json`,
`quality_final_ready9/summary.json`, and `comparison.json`. Raw observations are
retained on the board and locally; summaries are committed. Server 47 scoring
artifacts are under
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/expanded28_dist_opt_20260920`.

## Reproduction

Run on the board, with a new output directory each time:

```bash
cd /home/orangepi/deployments/expanded28_dist_opt_20260920
bash run_on_board.sh --output "$PWD/results/efficient" --save-observations
```

For strict upstream GMC with the tested lower-latency queue, use
`--gmc public --inflight 4 --dispatch round-robin`. For detection alone, add
`--detector-only --frames 2000`. These are throughput tests, not camera/display
latency tests. Rendering and encoding are excluded; per-frame JSON writing is
included when enabled. Do not run multiple benchmarks simultaneously.

Native shared-library rebuild:

```bash
bash /home/orangepi/ultralytics_yolov8_ziye/scripts/anti_uav/build_dist_detector_on_board.sh \
  /home/orangepi/deployments/expanded28_dist_opt_20260920/libanti_uav_detector.so
```

`scripts/anti_uav/evaluate_dist_optimized.py` scores actual board observations
against reviewed Video00009 annotations on server 47. It checks that every
detector result is identical to the original full RKNN run. GT is not read by
the board runtime. Server replay is only a quality comparison, **not board FPS**.

`scripts/anti_uav/summarize_dist_optimized.py` generates `comparison.json` from
the retained raw observations and summaries. The original baseline directory is
its `--baseline`; this directory is its `--root`.

## Thermal And Measurement Limits

This CM5 has a heatsink and **no fan**, confirmed by the user. PWM 255 is only a
control signal and does not establish cooling. A user-applied cold compress
temporarily lowered SoC temperature to about 53 C; full-load tests subsequently
returned to approximately 85 C. NPU fell from 1 GHz to 800 MHz, while separate
big-core frequency domains could fall as low as 408 MHz. Thermal protection was
never disabled, no frequency limits were raised, and no services were changed.

All full runs process 14,201 original 1920 x 1080 HEVC frames from Video00009;
steady FPS excludes the first 100 outputs. The file's 100 FPS is its source rate,
not model throughput. A 2,000-frame peak is not a sustained performance guarantee.
Runs have different thermal histories, including user-applied external cooling
during the session; their ratios are observed end-to-end gains,
not a controlled attribution to any single optimization.

Frame latency includes decode and waiting for an ordered result. Process startup
includes imports/model loading and is not power-on time. Stage durations overlap
and must not be summed or inverted to claim pipeline FPS.

Quality is measured at IoU 0.5 on reviewed frames, excluding uncertain/unreviewed
frames. ID-change counts are single-target continuity diagnostics, not standard
MOT IDSW/IDF1. Video00009 was used to diagnose/tune these settings, so this is not
evidence of generalization to a new independent test video.
