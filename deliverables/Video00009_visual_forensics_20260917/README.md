# Video00009 Visual Forensics and Continuity Candidate

Date: 2026-09-17. This work continues the earlier audit and parameter sweep by inspecting delivered video frames alongside crops of the original source. It does not replace the board service, model, or previously delivered videos.

## Visual Findings

The earlier explanation focused too much on small-target motion and confirmation thresholds. Visual inspection exposed a separate issue: dormant identities can steal detections from a valid, continuously observed identity, even for a large, high-confidence object.

1. **9.67-9.72 s, large clear drone:** the detector sometimes outputs a whole-drone box and a lower-confidence partial box. Both have developed mature tracks (IDs 1 and 2). At frame 970, ID 1 was observed on the previous frame and its match is admissible (cost 0.33618), but ID 2, last observed three frames earlier, wins with cost 0.32385. The visible object changes `1 -> 2 -> 1`, despite a detector score of 0.952 and no disappearance. This is not a small-target threshold failure. The duplicate detector boxes themselves are not removed by this experiment.
2. **119.50-119.64 s, one continuously visible small drone:** IDs 542, 545 and 538 alternate. At frame 11954, dormant ID 538 has not been observed for 0.75 seconds; it wins at cost 0.75133 over the current ID 542, observed 0.01 seconds earlier, at cost 0.83751. At frame 11964, dormant ID 545 wins over current ID 542 by only 0.00227 in cost. The shared assignment pool does not distinguish a continuous observation from revival of a stale identity.
3. **21.96/21.97 and 89.99/90.00 s:** the original images contain visible targets and the detector produces boxes. The renderer removes unconfirmed tracks. The 89.99-second detection has score 0.612 but only two hits, so it remains hidden. The candidate below still does not obtain a stable ID for this frame; the separate detection display correctly shows it as pending instead of silently omitting it.

Visual review included a whole-video overview at ten-second intervals, delivered detector/tracker frames at event times, clean-source consecutive-frame crops, and a 160-frame slow-motion comparison montage. This is targeted frame inspection, not a claim that every frame was manually reviewed. The original source SHA256 was verified as `2629efe7da36dfd079985344653a7a4ecaa25f651954e5258a53c50e960d6b86`.

## Candidate Change

An opt-in `Config::active_first` policy was added, default false. For high-score association it processes: previously observed confirmed tracks, then lost confirmed tracks, then tentative tracks. Every group retains the same spatial/class/cost admissibility tests. Lost-track recovery is still possible; it no longer competes before an admissible continuation of an active confirmed track. The low-score stage is unchanged and unused by the fixed conf=0.03 cache.

This is a targeted experimental state-management change, not a complete official BoT-SORT implementation or a universally safe solution for crowded/occluded targets. It may favor the wrong active object when an old object returns, so multi-target and reappearance validation remain necessary. The existing duplicate-detector and abrupt-motion limitations are not claimed solved.

## Full-Video Replay

Same 14,201 frames and exact detector boxes; metrics use 14,199 reviewed frames and IoU >= 0.5. No GT is supplied to the tracker. These are Video00009 diagnostic/validation results, not a new independent test.

| Method | TP | FP | Precision | Recall | Suppressed detector TPs | Continuous-GT ID changes | Adjacent TP-frame ID changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original delivered tracker | 5086 | 2677 | 65.52% | 63.72% | 216 | 153 | 35 |
| Correctness-fixed default control | 5086 | 2675 | 65.53% | 63.72% | 216 | 153 | 35 |
| Active-first, original match=0.92 | 5091 | 2686 | 65.46% | 63.78% | 211 | 99 | 12 |
| Active-first, match=0.96 | 5189 | 2872 | 64.37% | 65.01% | 113 | 49 | 9 |

The ID counts are the explicit single-GT diagnostics defined in the earlier report, NOT standard MOTChallenge IDSW/IDF1. A lower count can also result from incorrect merging. Fewer framewise switches do not prove correct long-term identity in every case.

With unchanged match=0.92, the active-first change fixes the representative large-target and small-target ID-hopping cases without relying on a relaxed cost threshold. With match=0.96 it also recovers the 21.96/21.97-second confirmed ID, but increases false positives by 195 compared with the original tracker. Precision falls by about 1.15 percentage points. The candidate still suppresses 113 detector true positives in confirmed-only output and is not ready to be called a complete fix.

The new default-off source replays byte-identically to the previous fixed-source default: displayed-track SHA256 `5ce5ab415c1067be47dafcee9087b3fa3b8d5a0dc91d890f8e202606d078db6b`.

## Video and Detection Display

Local video:

`/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/Video00009_visual_forensics_20260917/final_comparison_video/Video00009_tracker_visual_diagnosis_10x_slow.mp4`

Four columns: detector; original confirmed tracks; candidate confirmed tracks; candidate detections plus stable IDs. The final column retains unmatched/tentative detections in yellow as **DET pending**, without claiming a stable ID. This is a display separation, NOT improved tracker recall. It also retains detector false positives.

Across the full video, this display function preserves all 9,109 original detector boxes exactly; 1,048 have no confirmed observed ID in the candidate. Only these display records, not predicted boxes or GT boxes, are drawn. Unit tests verify that pending detections are not hidden, predicted/tentative tracks cannot claim stable IDs, and extra tracks cannot invent detections.

The montage contains source frames `[950,990)`, `[2190,2230)`, `[8985,9025)`, `[11940,11980)`, played at 10 FPS instead of source 100 FPS: 160 frames, 16 seconds, 1920x720. Diagnostic crops are centered on GT solely for inspection; GT boxes are not drawn, and GT does not affect association or detections. Frame previews and protocol/hash metadata are next to the video. Encoding was checked with ffprobe and a decoded preview.

## Implementation and Verification

- Native policy: `scripts/anti_uav/rknn_yolov8_native/detector_based_tracker.hpp`.
- Replay/traces: `scripts/anti_uav/audit_native_tracker_association.py`, now accepting `--active-first`, `--match-cost`, `--retain-raw`, and `--trace-frames`.
- Source/delivered-video contact sheets: `scripts/anti_uav/review_tracker_video_frames.py`.
- Montage and detection/ID separation: `scripts/anti_uav/render_tracker_forensic_clips.py`.
- Twelve native regression cases pass locally, including 900 brute-force assignment comparisons, an explicit dormant-competitor case, and separate-target/lost-track recovery under active-first. ASan/UBSan pass. Fourteen related Python tests pass.
- Baseline, control and candidate summaries are in this directory's `baseline/`, `default_control/`, `active_first092/`, and `active_first096/`. Representative native cost traces are in each selected candidate's `selected_cost_traces.json`; full replay files remain local and can be regenerated.

The native tracker is modified only behind an opt-in flag. No model retraining, camera pose input, board binary replacement or board FPS claim is involved. Code is synchronized by local Git push followed by server Git pull.

## Alternative Algorithms

Do not infer that the full BoT-SORT algorithm has these exact implementation defects. Its official implementation includes a different state-management/association pipeline and image-based camera motion compensation; ReID is optional. See [official BoT-SORT source](https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py).

For a controlled next comparison, use [standard ByteTrack](https://github.com/FoundationVision/ByteTrack) as the lightweight reference and [OC-SORT](https://github.com/noahcao/OC_SORT) as the observation-centric/nonlinear-motion candidate. Full BoT-SORT with image-based GMC is also relevant for camera movement. None of these alternatives has been benchmarked on this cache in this work; published pedestrian/other-domain benchmarks do not establish the winner for tiny grayscale UAVs. ReID benefit at 4-8 input pixels is an engineering uncertainty, not something to assume from generic person-tracking benchmarks.
