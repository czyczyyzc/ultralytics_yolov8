# Experimental Motion-Aware Balanced Tracker

This is an independent algorithm built from the native Dist matrix/LAP primitives.
It is NOT behavior-equivalent public Dist, full OC-SORT, or a complete IMM filter.
The default native pipeline remains `--tracking dist`. Existing detector models,
legacy tracker ABI, and detector confidence/NMS are unchanged.

## Algorithm

- Primary association combines uncertainty-normalized center innovations, covariance-volume penalties, size agreement, motion direction and IoU. Zero IoU is allowed within bounded motion/shape gates; there is no post-failure nearest-neighbor rescue.
- Smooth-motion and bounded-maneuver hypotheses are scored together in the primary cost. The scores and GMC reliability are engineering scores, not calibrated posterior probabilities.
- IoU receives more weight for large boxes; tiny boxes receive more motion-distance weight. Association gates constrain relative changes, not absolute box area. Large detections are not discarded.
- Kalman propagation uses source time deltas. Acceleration noise has a pixel-independent floor and adapts to observed residuals. Recent real observations update velocity with localization-uncertainty weighting; both pending and confirmed tracks are predicted.
- GMC still estimates the same warps. The added quality score is `min(1, inliers/24) * inlier_ratio` for accepted fits, otherwise zero. Failed identity transforms and confidently estimated stationary transforms are therefore distinguishable. Invalid GMC increases positional uncertainty instead of asserting zero camera motion.
- The balanced preset requires 3 matched observations in the last 4 processed frames, plus average observed confidence at least the birth threshold. Confirmed tracks survive short gaps internally; outputs always reference current detector observations. No predicted/pending box display or ID remapping is used.
- Ambiguous row/column assignments are rejected, including new births for those ambiguous observations. Expiry is in seconds, not a fixed number of processed frames.

## Build and Native Inference

On the board, build a NEW directory with the existing OpenCV/OpenSSL development environment:

```bash
bash scripts/anti_uav/dist_native/build.sh /home/orangepi/deployments/motion_aware_20261008
```

This builds `libdist_tracker.so`, `libmotion_tracker.so`, the quality-capable
`libdist_gmc.so`, and `anti_uav_dist_native`. It does not generate a detector
RKNN or install/replace the existing detector shared library.

Invoke the new executable using a platform-compatible, verified detector model/library:

```bash
NEW=/home/orangepi/deployments/motion_aware_20261008
"$NEW/anti_uav_dist_native" \
  --tracking motion \
  --tracker-library "$NEW/libmotion_tracker.so" \
  --gmc-library "$NEW/libdist_gmc.so" \
  --detector-library /absolute/path/to/verified/libanti_uav_detector.so \
  --model /absolute/path/to/platform-compatible-detector.rknn \
  --video /absolute/path/to/source.mp4 \
  --output /absolute/path/to/new-output-directory \
  --workers 3 --inflight 3 --conf 0.03 --iou 0.45 --save-observations
```

RK3576 needs two workers/core masks rather than the RK3588 three-core settings.
All actual inference/association/GMC remains C++. Python/ctypes scripts below
are OFFLINE verification/rendering only. A quality-capable GMC library is
required; an old library missing `gmc_quality` intentionally fails rather than
silently pretending motion quality is known.

Replacing the shared library alone does NOT select the new algorithm: the
library retains the legacy Dist symbols. Use the rebuilt executable and the
explicit `--tracking motion` flag.

Live camera motion mode requires valid monotonically increasing driver frame
timestamps. File mode uses `index/source_fps` and therefore assumes CFR input.
Dropped live frames change `dt`; processed-frame indices do not substitute for
capture time. The camera's driver timestamp is not a verified exposure time.

## Configuration ABI

`motion_defaults(fps)` in `motion_tracker.hpp` and `DEFAULTS` in
`motion_native_runtime.py` use this identical ordered 14-value configuration:

| Position | Meaning | Default |
| ---: | --- | ---: |
| 0 | high detector observation threshold | 0.03 |
| 1 | low detector observation threshold | 0.01 |
| 2 | birth threshold / confirmation mean-confidence threshold | 0.10 |
| 3 | expiry seconds | 1.0 |
| 4 | nominal source FPS | source FPS |
| 5 | localization standard-deviation floor, source pixels | 1.5 |
| 6 | baseline acceleration standard deviation, px/s2 | 180 |
| 7 | unknown-camera motion uncertainty, px/s | 1500 |
| 8 | innovation search budget, px/s | 3000 |
| 9 | maximum center-innovation radius, source pixels | 240 |
| 10 | squared normalized-innovation gate | 16 |
| 11 | confirmation observations | 3 |
| 12 | confirmation window, processed frames | 4 |
| 13 | ambiguous-cost margin | 0.03 |

The radius also includes a three-sigma localization allowance. Units refer to
original-frame coordinates, not network input pixels; defaults were evaluated
on 1920x1080 sources and need calibration for substantially different optics,
resolution, motion or detector error. Width/height observation noise uses 5%
of the corresponding detection dimension with a floor.

Native overrides use `--motion-params` followed by 14 comma-separated values;
nominal FPS must equal the source FPS. Summaries save the actual ordered values.
For example, the faster-ID 2-of-3 profile at 30 FPS is:

```text
--motion-params 0.03,0.01,0.10,1,30,1.5,180,1500,3000,240,16,2,3,0.03
```

The balanced preset needs at least three detections before emitting a new ID,
so earliest identity confirmation is the third observation. This delay is
separate from first detector-result latency. Do not bypass confirmation to
make a visualization appear more complete.

## Offline Verification

```bash
python -m pytest tests/test_motion_native_tracker.py \
  tests/test_cached_tracker_video.py tests/test_dist_tracker_cache.py -q
python scripts/anti_uav/track_cached_native_dist_video.py \
  --tracker-kind motion --source /absolute/path/to/CFR.mp4 \
  --detector-dir /absolute/path/to/complete-detector-cache \
  --tracker-library /absolute/path/to/libmotion_tracker.so \
  --baseline-dir /absolute/path/to/original-tracker-cache \
  --output /absolute/path/to/new-replay-output
```

`--baseline-dir` verifies unchanged detections and identical GMC matrices.
`--cached-gmc` explicitly reuses a quality-aware GMC cache for offline ablations;
it is NOT a native live-inference backend. Optional `--motion-config` uses the
complete named configuration JSON and must have the correct source FPS.

2026-10-08 verification: 23 tests passed on macOS and the 47 server; original
Dist IDs/observation indices were exactly reproduced across 1800 frames per
model; the native quality ABI was checked on textured stationary and featureless
images. Linux native pipeline/GMC C++17 syntax compilation passed. These are
NOT RK board FPS, live-camera accuracy, or full RKNN pipeline acceptance tests.

## Results and Limits

See `deliverables/motion_aware_tracker_20261008/RESULTS_ZH.md` and its raw JSON.
The 000002 failure segment is repaired without detector changes or fabricated
boxes. Video00009 validation identity continuity improves, but FP increases;
do not automatically replace the production tracker. New hyperparameters were
developed using these videos, so the results are not independent generalization
evidence. Identity diagnostics are not formal MOT IDSW/IDF1/HOTA.

All runtime C++/libraries must be rebuilt for the board architecture. The 47
server's shared libraries are x86_64 and must not be copied to an arm64 board.
Existing baseline code and earlier trial artifacts are retained for audit.
Licenses remain those of the native Dist-derived code and vendored LAP sources.
