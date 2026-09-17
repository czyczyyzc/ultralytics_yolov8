# Parameter-Only Native Tracker Sweep

Date: 2026-09-17. All 14,201 Video00009 frames replayed; 14,199 reviewed frames scored at IoU >= 0.5. Same immutable conf=0.03 detector cache and correctness-fixed native tracker as the preceding audit. No low-score supplementation, camera-motion compensation, predicted output or confirmed-first policy. No model inference, retraining, GPU use or deployment change was needed.

## Finding

Parameter mismatch is an important contributor, especially the association-cost limit. The earlier structural audit did not establish the limit of parameter tuning. A sweep of 24 configurations now shows materially fewer diagnostic identity changes when association is relaxed, with an explicit false-positive tradeoff.

| Change from fixed-code baseline | TP | FP | Precision | Recall | F1 | Continuous-GT ID changes | Adjacent TP-frame ID changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: match 0.92, birth 0.10, hits 3, buffer 1 s | 5086 | 2675 | 65.53% | 63.72% | 64.61% | 153 | 35 |
| Match cost 0.96 only | 5181 | 2859 | 64.44% | 64.91% | 64.67% | 94 | 21 |
| Match cost 0.98 only | 5215 | 2961 | 63.78% | 65.33% | 64.55% | 54 | 11 |
| Match cost 0.99 only | 5232 | 3071 | 63.01% | 65.55% | 64.26% | 51 | 13 |
| Hits 2 only | 5162 | 2820 | 64.67% | 64.67% | 64.67% | 193 | 54 |
| Birth 0.03 only | 5085 | 2754 | 64.87% | 63.71% | 64.28% | 154 | 35 |
| Birth 0.03 and hits 1 | 5302 | 3806 | 58.21% | 66.42% | 62.05% | 305 | 146 |

Raising the maximum association cost allows weaker matches; it does not raise the detector confidence and does not mean more stringent matching. The first and second cost limits were set together, but this >=0.03 input cache has no low-stage detections, so only the first-stage change can affect these results. These are this custom tracker's costs, not interchangeable with another BoT-SORT implementation's thresholds.

Match=0.96, birth=0.10, hits=3, buffer=1 s is a reasonable next cross-video validation candidate. It recovers 95 true positives but adds 184 false positives versus the fixed-code baseline, and F1 rises only about 0.06 percentage points. Match=0.98 further reduces the identity-change diagnostic but increases false positives. No candidate is declared universally better or enabled by default.

Lowering confirmation from three hits to two recovers boxes but makes more unstable identities visible. Removing the birth/confirmation gates entirely retains all detector boxes, but does not solve tracking identity continuity. It produces 305 continuous-GT ID changes in this diagnostic, compared with 153 at baseline.

## Limits

- Identity-change counts use exactly the earlier audit definition: compare successive correctly localized observations while a GT target is continuously annotated; adjacent counts require consecutive correctly matched frames. They are NOT standard MOTChallenge IDSW/IDF1. Fewer changes can also reflect incorrect merging and must not be treated as a complete identity-quality evaluation.
- Video00009 is a model-selection/diagnostic video. Selecting a preset on it does not constitute independent test improvement. Validate on other videos, occlusion/reappearance and multi-target or clutter cases before deployment.
- The hard spatial gate still rejects candidates when IoU < 0.01 and normalized center distance > 3. Increasing the cost limit cannot undo this rejection. Camera-motion compensation and motion uncertainty remain relevant for larger abrupt displacements.
- The sweep is not exhaustive. Match costs tested: 0.92, 0.94, 0.96, 0.98, 0.99, crossed with birth 0.10/0.03 and hits 3/2 at buffer=1 s (20 runs). Additional controls: birth=0.03/hits=1; baseline with buffer=0.3, 0.5, 2 s (four runs).

## Reproduction

Script: `scripts/anti_uav/sweep_native_tracker_parameters.py`.

Results, provenance and all 24 configurations: `sweep.json` in this directory. The script verifies detector/GT hashes, frame coverage/timestamps, the previously audited fixed-code baseline, and equality to detector counts when birth/confirmation suppression is disabled. Three additional unit tests cover the identity diagnostic, tracking gaps, excluded frames and wrong localizations.

From the local repository root:

```bash
python3 scripts/anti_uav/sweep_native_tracker_parameters.py \
  --detector-dir deliverables/Video00009_expanded28_20260917 \
  --tracker-dir deliverables/Video00009_expanded28_tracker_20260917 \
  --approved-manifest .codex_work/video00009_tracker_diagnosis/approved_manifest.json \
  --coco .codex_work/video00009_tracker_diagnosis/annotations.json \
  --output deliverables/Video00009_tracker_parameter_sweep_fresh
```

The server repository receives the script and report through local Git push followed by server Git pull. For server replay, use the original run/GT paths documented in `../Video00009_tracker_deep_audit_20260917/REPORT.md` and a fresh output directory. The saved sweep was run locally against the same native C++ tracker, not on the RK board; no FPS inference is made.
