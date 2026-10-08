# Motion-Aware Tracker and Lossless Observation Output

This is an independent algorithm built from the native Dist matrix/LAP primitives.
It is NOT behavior-equivalent public Dist, full OC-SORT, or a complete IMM filter.
The default native pipeline remains `--tracking dist`. Existing detector models,
legacy tracker ABI, and detector confidence/NMS are unchanged.

## Algorithm

- Primary association combines uncertainty-normalized center innovations, covariance-volume penalties, size agreement, motion direction and IoU. Zero IoU is allowed within bounded motion/shape gates; there is no post-failure nearest-neighbor rescue.
- Smooth-motion and bounded-maneuver hypotheses are scored together in the primary cost. The scores and GMC reliability are engineering scores, not calibrated posterior probabilities.
- IoU receives more weight for large boxes; tiny boxes receive more motion-distance weight. Association gates constrain relative changes, not absolute box area. Large detections are not discarded.
- Kalman propagation uses source time deltas. Acceleration noise has a pixel-independent floor and adapts to observed residuals. Recent real observations update velocity with localization-uncertainty weighting; both pending and confirmed tracks are predicted. Uncertain GMC also propagates into observation-history variance, avoiding false velocity updates when a failed camera estimate later recovers.
- GMC still estimates the same warps. The added quality score is `min(1, inliers/24) * inlier_ratio` for accepted fits, otherwise zero. Failed identity transforms and confidently estimated stationary transforms are therefore distinguishable. Invalid GMC increases positional uncertainty instead of asserting zero camera motion.
- Camera uncertainty adapts to recent accepted warp speed and to residual motion from already-confirmed continuations. The search radius incorporates Kalman position uncertainty and recent observed target extent. The configured 240-pixel radius remains the floor for the bounded cap; large/clipped targets may use up to twice their recent diagonal, while small stationary distractors do not inherit that relaxation.
- Confirmed tracks are associated before tentative tracks, under the same gates and ambiguity checks. Large/clipped targets use recent GMC-warped observed extents for overlap/size agreement, not lagging predicted box geometry.
- The balanced preset requires 3 matched observations in the last 4 processed frames, plus average observed confidence at least the birth threshold. Confirmed tracks survive short gaps internally; outputs always reference current detector observations. No predicted box display or ID remapping is used.
- Every current detector observation is retained in a separate identity-status output, including weak, pending and ambiguous observations. These are not counted as confirmed tracks. The optional 16-value profile assigns a candidate ID from the first eligible observation and preserves it on promotion; this ID is explicitly tentative. Observations with unresolved assignment or overlapping-new-candidate ambiguity expose a null ID instead of a fabricated identity.
- A new tentative box strongly nested in an already-established current target is retained but marked ambiguous. Two previously confirmed targets are never merged by this rule.
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

The executable also supports `--gmc-mode unavailable` for a native no-GMC
control. It skips loading the GMC library and skips the GMC worker, supplies an
identity transform, and retains the configured unknown-camera uncertainty.
Use this for controlled experiments, not as the default deployment: labelled
Video00009 regression testing showed worse continuous identity stability without
GMC. Production guidance is quality-gated GMC with uncertainty fallback on
failed frames.

Replacing the shared library alone does NOT select the new algorithm: the
library retains the legacy Dist symbols. Use the rebuilt executable and the
explicit `--tracking motion` flag.

Live camera motion mode requires valid monotonically increasing driver frame
timestamps. File mode uses `index/source_fps` and therefore assumes CFR input.
Dropped live frames change `dt`; processed-frame indices do not substitute for
capture time. The camera's driver timestamp is not a verified exposure time.

## Configuration ABI

The legacy ABI remains the identical ordered 14-value configuration used by
`motion_defaults(fps)` in `motion_tracker.hpp` and `DEFAULTS` in
`motion_native_runtime.py`:

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

The optional causal-candidate profile accepts two appended values:

| Position | Meaning | Default |
| ---: | --- | ---: |
| 14 | candidate allocation threshold | 0.03 |
| 15 | emit tentative candidate IDs | 1 |

The radius also includes a three-sigma localization allowance and a
`sqrt(nis_gate * largest_position_innovation_variance)` allowance. Units refer to
original-frame coordinates, not network input pixels; defaults were evaluated
on 1920x1080 sources and need calibration for substantially different optics,
resolution, motion or detector error. Width/height observation noise uses 5%
of the corresponding detection dimension with a floor.

Native overrides use `--motion-params` followed by 14 or 16 comma-separated values;
nominal FPS must equal the source FPS. Summaries save the actual ordered values.
For example, the faster-ID 2-of-3 profile at 30 FPS is:

```text
--motion-params 0.03,0.01,0.10,1,30,1.5,180,1500,3000,240,16,2,3,0.03
```

The balanced preset needs at least three detections before confirming a new ID.
With the 16-value candidate profile, the measured box may carry a tentative ID
from the first observation and keeps the same ID after confirmation. This does
not turn first-frame output into confirmed tracking. Ambiguous observations
remain ID-less.

## Consumer Output Contract

`motion_update` retains its confirmed-only `[id, detection_index]` ABI. After
each successful update, call `motion_observations(handle, triples, capacity)`;
it returns one ordered `[id_or_zero, detection_index, status]` per input
detection, even for inputs below the tracker's low threshold. The call is
read-only. Capacity must cover all detections; an undersized buffer returns
`-1`, rather than silently truncating outputs.

| Status | JSON value | Identity | Meaning |
| ---: | --- | --- | --- |
| 0 | `unassigned` | null | No plausible association/birth evidence |
| 1 | `pending` | null or positive candidate ID | Tentative track awaiting confirmation |
| 2 | `confirmed` | positive ID | Confirmed identity matched this current detection |
| 3 | `ambiguous` | null | Multiple plausible associations; no forced identity |
| 4 | `below_low` | null | Detection retained but excluded from track association |

The native executable always computes this contract for `--tracking motion`.
`--save-observations` writes the complete `observations` array into each JSONL
frame. Its `box` and `score` are the original current detector measurement;
`predicted` is always false. `displayed_tracks` remains confirmed-only for
backward-compatible evaluation. Visualization/platform integrations that must
not swallow detector boxes must consume `observations`, NOT only
`displayed_tracks`. Draw candidate observations as `ID n?` and unresolved
observations as `DET`/`UNCERTAIN`. This is explicit observation retention, not
post-hoc ID repair. Never present a candidate ID as confirmed.

Video/camera inference, observation-status generation, GMC and association all
remain C++. The Python adapter mirrors this native contract for offline audit.
Internal Kalman predictions can bridge gaps, but no box is emitted on a frame
with no detector measurement. Long occlusion/expiry can lead to a new ID; no
appearance ReID is claimed.

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

`--gmc-mode disabled` is an OFFLINE image-coordinate ablation: it does not
decode/estimate background motion, supplies an identity warp and quality zero,
and sets `unknown_gmc_speed_px_s=0`. The tracker then learns total image motion
without adding the uncertainty of a failed compensation to every velocity
observation. This is different from an enabled GMC temporarily failing.
Detector provenance is still verified against `--baseline-dir`; GMC equality is
deliberately not required in disabled mode. Do not combine disabled mode with
`--cached-gmc`. This flag is not currently a native board executable option.

`--gmc-mode unavailable` supplies identity warp and quality zero while retaining
configured unknown-camera uncertainty. It isolates removal of the GMC estimate
from removal of the uncertainty allowance. The Python replay supports this
control, and the native executable skips all GMC loading/computation in this
mode. On 000002 it preserves the primary ID through the former 15.8-second
failure, but on labelled Video00009 it causes substantially more within-segment
ID changes. A single clip therefore does not justify globally removing GMC.

The immediate-ID experiment uses `birth=.03`, `confirmation_hits=1`, and
`confirmation_window=1`. It assigns a new ID on the first eligible detection,
not a reliable identity certified by extra evidence. Ambiguous associations
still return null ID. The default balanced profile is NOT changed.
`compare_motion_gmc_ablation.py` audits both modes/policies against identical
detector inputs, verifies balanced + GMC reproduces the previous regression,
and reports top-score observation diagnostics separately from formal GT metrics.
See `deliverables/clip_000002_gmc_id_ablation_20261008/RESULTS_ZH.md`.

2026-10-08 verification: 54 tests passed on macOS; 54 tests plus 4 subtests
passed on the 47 server. Linux native pipeline C++17 syntax compilation passed.
These are NOT RK board FPS, live-camera accuracy, or full RKNN pipeline
acceptance tests.

## Results and Limits

See `deliverables/clip_000002_causal_global_20261008/RESULTS_ZH.md`. The former
000002 failure at frames 473-475 is repaired without detector changes or
fabricated boxes: both P3 and P2+P3 keep the main target at confirmed ID 1.
All 547 P3 and 666 P2+P3 detector observations are retained. P2+P3's nested
partial duplicate at frame 474 is retained as `UNCERTAIN`, while the complete
target remains ID 1. There are no adjacent single-confirmed-target ID changes
in either 1800-frame replay.

On labelled Video00009, quality-gated GMC has zero continuous-GT ID changes;
the no-GMC control has 17 when tentative IDs are included and one on
confirmed-only output. With quality-gated GMC, confirmed-only TP/FP/FN is
5296/3087/2686 (precision 63.18%, recall 66.35%). All measured observations
remain detector-equivalent at TP/FP/FN 5302/3806/2680. These are fixed-cache
diagnostics, not formal MOT IDSW/IDF1/HOTA, and the 000002 clip has no complete
identity GT. The algorithm does not claim identity recovery after expiry or
long disappearance.

Use `render_cached_tracker_result_video.py --show-observations` to visualize the
new contract. Omit that flag to inspect confirmed-only tracking. Use
`audit_motion_observations.py` with detector/tracker/baseline directories and an
optional approved manifest to report measured-box coverage and confirmed-only
metrics separately. The audit verifies matching detector/GMC inputs and cache
hashes. `MOTION_TRACE` is an optional compile-time diagnostic; its frame range is
set with `MOTION_TRACE_BEGIN`/`MOTION_TRACE_END`. Normal builds contain no trace.

All runtime C++/libraries must be rebuilt for the board architecture. The 47
server's shared libraries are x86_64 and must not be copied to an arm64 board.
Existing baseline code and earlier trial artifacts are retained for audit.
Licenses remain those of the native Dist-derived code and vendored LAP sources.
