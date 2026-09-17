# Video00009 Native Tracker Audit

Date: 2026-09-17. This is a diagnostic replay, not a new model release or a board FPS benchmark.

## Conclusion

The missing boxes and frequent identity changes are real. Detector confidence, track birth, track confirmation, and association are separate gates. Fixing two confirmed C++ bugs does not resolve the main fast-small-target association failure. Simply supplying lower-score detections is not a net improvement on this video. No experimental configuration has been enabled on the board.

## Controlled Protocol

- All 14,201 source frames were replayed sequentially at source timestamps (100 FPS).
- Evaluation includes 14,199 reviewed frames, 7,982 GT boxes; uncertain frames 9362 and 9363 are excluded. Frame numbers in this report are zero-based.
- Per-frame matching uses IoU >= 0.5. Precision/recall below are box counts, not mAP or tracking IDF1.
- Model: 28-video Frozen-P3 + Add-on P2, PT FP32, 960x544, detector conf=0.03, NMS IoU=0.45. This does not measure RKNN quantization or board throughput.
- Model SHA256: `61e9a3669f964e59e96e9b2a24bf7705b945e184ed5a0628be33ee44d8b4bb59`.
- Original detector cache SHA256: `c1edc4cd5b13756389a33351d631066cc7c4bb9b1c19420ceae33233112ddc29`.
- Baseline replay matches every original raw tracker record exactly, including IDs, boxes, hits and age. Ground truth is used only for scoring and retrospective diagnostics, never as tracker input.
- Video00009 has previously been used for checkpoint selection. These are diagnostic/validation results, not a new independent held-out test claim.

## Confirmed Causes

### Detection Is Not Confirmation

The detector exports boxes at conf >= 0.03. The current native tracker additionally requires score >= 0.10 to create an unmatched new track and three cumulative hits before the renderer displays it. A score between 0.03 and 0.10 can update an existing compatible track, but cannot start a new one. Three hits means cumulative hits, not necessarily three consecutive frames.

The renderer displays only confirmed, currently observed tracks. `prediction_grace_sec=0` means no predicted boxes are drawn through missed observations. `track_buffer_sec=1` retains an ID internally; it does not mean that a box remains visible for one second.

There are 915 frames with a detector box but no displayed track. They must not all be called genuine missed targets: many detections are false positives. On reviewed GT, the tracker suppresses 216 detector true positives: 192 remain tentative and 24 fail association and cannot be born below the birth threshold. It also removes 1,129 detector false positives.

### Fast Motion Breaks Identity Association

Association uses predicted/last-box IoU, size-normalized center distance, and a Kalman distance term. The C API used here supplies no camera motion transform or appearance embeddings. The current implementation is a lightweight custom RK-BoT-SORT-style tracker, not a full image-based GMC/ReID pipeline.

Concrete continuous-target sequence:

| Frame | Time | Detector score | Raw ID | Hits | Displayed |
|---:|---:|---:|---:|---:|---|
| 2195 | 21.95 s | 0.173 | 1 | 1300 | Yes |
| 2196 | 21.96 s | 0.331 | 4 | 1 | No |
| 2197 | 21.97 s | 0.315 | 4 | 2 | No |
| 2198 | 21.98 s | 0.393 | 4 | 3 | Yes |
| 2207 | 22.07 s | 0.401 | 5 | 1 | No |
| 2208 | 22.08 s | 0.107 | 6 | 1 | No |
| 2209 | 22.09 s | 0.196 | 7 | 1 | No |
| 2210 | 22.10 s | 0.383 | 8 | 1 | No |
| 2211 | 22.11 s | 0.307 | 9 | 1 | No |
| 2212 | 22.12 s | 0.081 | None | - | No |

At frame 2196 the old confirmed ID's association cost is 0.9591, over the 0.92 limit. A new ID is created despite a correct detector box. At frame 2197 the old ID is admissible (cost 0.9053), but the tentative ID is a much closer match (cost 0.4236), so the tentative ID wins the shared assignment pool and remains hidden.

GT width is only about 10-12 pixels in the original 1920x1080 frames. Adjacent GT center displacement is approximately 28.6 pixels at frame 2196, 57.3 at 2208 and 94.5 at 2209. This is consistent with image-space motion far exceeding target dimensions; it does not by itself identify how much comes from camera versus target motion. The current small-box motion model/gating is not robust to this sequence.

Of the original 216 suppressed true-positive frames, 139 have existing tracks but no admissible association, and 15 have an admissible confirmed competitor. These are diagnostic subsets and should not be summed with the tentative/birth breakdown.

### Two Correctness Bugs Fixed

1. Inadmissible edges entered Hungarian optimization and were rejected only afterward. They could displace valid assignments. Regression example: costs `[[0.10, 0.93], [0.20, 10.0]]`, limit 0.92, gave `[-1, 0]` instead of `[0, -1]`. Non-finite and over-limit costs are now masked before solving. The unmatched objective/thresholds are otherwise unchanged.
2. Expiration happened after association, allowing a detection arriving after the time buffer to reactivate an expired ID. Expired tracks are now removed before prediction/association, after timestamp normalization.

Both were reproduced before the fix. The first affects six association decisions in the original video replay. The combined fix removes two false positives but does not recover additional true positives here. The expiration regression matters for gaps/sparse calls even though it has no additional measured effect on this regularly sampled video.

## Results

All tracker rows use high=0.03, low=0.01, birth=0.10, min_hits=3, match=0.92, buffer=1 s and grace=0. The optional confirmed-first experiment gives confirmed tracks first access to high-score detections, then tentative tracks, leaving the low stage otherwise unchanged. It is NOT a full BoT-SORT state-machine replacement.

| Method | TP | FP | FN | Precision | Recall | F1 | Continuous-GT ID changes | Adjacent TP-frame ID changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector only, conf=0.03 | 5302 | 3806 | 2680 | 58.21% | 66.42% | 62.05% | N/A | N/A |
| Original tracker | 5086 | 2677 | 2896 | 65.52% | 63.72% | 64.60% | 153 | 35 |
| Two correctness fixes | 5086 | 2675 | 2896 | 65.53% | 63.72% | 64.61% | 153 | 35 |
| Fixes + confirmed-first | 5093 | 2690 | 2889 | 65.44% | 63.81% | 64.61% | 133 | 31 |
| Fixes + low-score input | 5095 | 2848 | 2887 | 64.14% | 63.83% | 63.99% | 152 | 36 |
| Fixes + confirmed-first + low-score input | 5104 | 2861 | 2878 | 64.08% | 63.94% | 64.01% | 130 | 30 |

Identity counts are explicit single-GT diagnostics, NOT standard MOTChallenge IDSW/MOTA/IDF1. A matched GT observation is assigned its best-IoU confirmed track. The continuous-GT count compares successive matched observations only when every intervening reviewed frame has a GT target; the adjacent count requires consecutive matched frames. These counts may omit identity failures during undetected periods, and lower counts alone do not prove globally better tracking. Full events are in `final_*/identity_change_events.json`.

### Low-Score Ablation

47-server GPU 6 re-inferred the same video/model at conf=0.01, batch 32, FP32. It completed in 88.67 seconds (offline GPU inference/cache generation, NOT RK board FPS). Every high-score result was checked against the original within 1e-4; the supplemented cache retains the EXACT original high-score boxes and adds 1,260 boxes in [0.01, 0.03). No GT was used. The original visualization's conf=0.03 cache could not exercise the configured low-score stage.

The main table counts all confirmed observed tracker outputs, including associated low-score boxes in the last two rows. If displayed boxes are additionally restricted to score >= 0.03:

| Method | TP | FP | Precision | Recall |
|---|---:|---:|---:|---:|
| Fixes + low-score input | 5085 | 2704 | 65.28% | 63.71% |
| Fixes + confirmed-first + low-score input | 5094 | 2718 | 65.21% | 63.82% |

Thus simply restoring low-score input does not solve the loss/identity problem, even with the original display confidence held fixed.

## Release Decision

- The two correctness fixes are retained in the source. No weights, thresholds, board service or previously delivered visualization have been overwritten.
- `Config::confirmed_first` is experimental and defaults to false. Its modest identity benefit comes with more false positives; do not silently enable it based only on Video00009.
- Detection output and track identity should be treated separately: retain conf>=0.03 detections as detections/pending targets, and attach stable IDs only after association/confirmation. This prevents the display from implying detector failure, but is not itself an identity-stability fix. This display change is a recommendation, not claimed as implemented here.
- To improve actual identity continuity, next validate image-based camera-motion compensation and uncertainty-aware motion association for fast tiny targets. Validate recovery and false-ID attachment across multiple videos before enabling looser motion gates. Pose input is not inherently required for image-based motion compensation. No new motion policy is claimed tested in this report.

## Files and Reproduction

Server repository: `/mnt/chenziye/codes/ultralytics_yolov8`.

Run root: `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916`.

Weights: run root + `/expanded_28/training_addon/p2/weights/best.pt`.

Original detector cache: run root + `/Video00009_expanded28_visual_full_20260917`.

Original tracker cache: run root + `/Video00009_expanded28_tracker_full_20260917`.

Supplemented detector cache: run root + `/Video00009_tracker_low_score_cache_20260917`.

GT task: `/mnt/andrew/anti_uav_model_refinement/data/approved_tasks/task-30298c02994f74ca6ddddb72d60b18570f34ca80f4fd6f4a46a6493d51904468`.

Local report and final summaries: `/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/Video00009_tracker_deep_audit_20260917`.

Native source/tests: `scripts/anti_uav/rknn_yolov8_native/detector_based_tracker.hpp`, `detector_based_tracker_test.cpp`, `tracker_regression_probe.cpp` and `tracker_trace_api.cpp`.

The archived `baseline.hpp` in this report directory contains the original behavior plus read-only diagnostic instrumentation. Its replay was verified against the original saved raw tracks for all frames. It is evidence, not a replacement production header. To reproduce the baseline trace build, copy it to an isolated directory as `detector_based_tracker.hpp`, copy the current `tracker_c_api.cpp` and `tracker_trace_api.cpp` alongside it, and compile `tracker_trace_api.cpp` with `-DRK_TRACKER_LEGACY_TRACE`.

Example fixed-code replay on the server, from its repository directory:

```bash
g++ -O3 -std=c++17 -shared -fPIC scripts/anti_uav/rknn_yolov8_native/tracker_trace_api.cpp -o /tmp/rk-tracker-audit-fixed.so
R=runs/anti_uav/native_fullpool_14_vs_28_20260916
G=/mnt/andrew/anti_uav_model_refinement/data/approved_tasks/task-30298c02994f74ca6ddddb72d60b18570f34ca80f4fd6f4a46a6493d51904468
.venv_gray_synthesis/bin/python scripts/anti_uav/audit_native_tracker_association.py \
  --detector-dir "$R/Video00009_expanded28_visual_full_20260917" \
  --tracker-dir "$R/Video00009_expanded28_tracker_full_20260917" \
  --approved-manifest "$G/manifest.json" --coco "$G/coco/annotations.json" \
  --library /tmp/rk-tracker-audit-fixed.so --name fixed \
  --output "$R/tracker_audit_fixed_fresh"
```

Use a fresh output directory for every run. Add `--confirmed-first` only for the experimental priority arm. Add `--supplement-low-dir "$R/Video00009_tracker_low_score_cache_20260917"` only for the low-score arm. The audit rejects modified original high-score boxes, mismatched source/model hashes, partial frame caches and mismatched GT hashes.

Validation: ten C++ regression cases, including 900 randomized small assignment problems checked against brute force; eight Python unit tests. The native tests also pass AddressSanitizer and UndefinedBehaviorSanitizer on the Mac. The Linux g++ regression and cross-platform replay results are recorded separately in `VERIFICATION.md`.
