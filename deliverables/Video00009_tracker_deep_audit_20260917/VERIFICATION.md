# Verification Record

Verified on 2026-09-17 against code/report revision `7ffcc0b` (fix revision `15f98b9`).

## Regression Checks

- Mac clang++: ten native C++ test cases pass, including 900 random assignment matrices checked against an exhaustive oracle.
- Mac AddressSanitizer + UndefinedBehaviorSanitizer build: all ten cases pass, no reported sanitizer errors.
- Linux g++ on 47 server: all ten native cases pass.
- Python on Mac and Linux: four cache-suppression scoring tests and four low-score-cache integrity tests pass.
- The standalone probe failed both checks before the fixes and passes both afterward: inadmissible assignment edges cannot steal a match, and expired tracks cannot revive.
- Baseline Mac replay reproduces all 14,201 original raw tracker records exactly, including boxes, identities and lifecycle metadata.

## Cross-Platform Replay

Two fixed-source Linux replay arms were compared with the Mac results. Their entire displayed-track JSONL files are byte-identical, not just their rounded metrics.

| Arm | Frames | TP | FP | FN | Continuous-GT ID changes | SHA256 of tracks.jsonl |
|---|---:|---:|---:|---:|---:|---|
| Correctness fixes only | 14201 | 5086 | 2675 | 2896 | 153 | `5ce5ab415c1067be47dafcee9087b3fa3b8d5a0dc91d890f8e202606d078db6b` |
| Fixes + confirmed-first + low input | 14201 | 5104 | 2861 | 2878 | 130 | `161121f315462b63c236422be266bf60b32dd75fa48e51e6e08e3b6ac2511d33` |

Linux output root:

`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/Video00009_tracker_deep_audit_linux_20260917`

Subdirectories: `fixed/` and `priority_low/`; each contains `summary.json`, `tracks.jsonl`, `identity_change_events.json`, and `missed_tp_events.json`.

Linux trace library SHA256: `d56ccc4c8c3216be630fb827d1090060c1d4f93ff0a7a88f773f52de5e239871`.

The main report defines the identity diagnostic precisely. It is not a standard IDSW metric and is not a claim that tracking has been solved. No board binary, running service, inference thresholds, model weights, or previous visualization was replaced. Local changes were pushed first, then the server pulled the same commits from a Git bundle because direct server GitHub SSH access is unavailable.
