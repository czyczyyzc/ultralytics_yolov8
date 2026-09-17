# Dist-Tracker Public-Code Evaluation on Video00009

Date: 2026-09-17. This is an offline validation experiment, not a board deployment or an independent held-out benchmark. The detector, weights, source video, original detection boxes and annotations are unchanged.

## Finding

The public repository with sparse optical-flow camera motion compensation (GMC), adapted to our low-confidence detections, substantially improves identity continuity on this video. Recall improves, but false positives increase. It is not uniformly better and it has not been benchmarked on RK hardware.

The repository is intended for UAVs, but the checked-out revision `396c359e1aa8be4fd5e81a02626cb1ee3867cf7c` does not contain the paper's FLIT L2-IoU association in the executed path. `ultralytics/cfg/default.yaml:128` selects `botsort.yaml`; `trackers/track.py` maps it to BOTSORT; `bot_sort.py:213` calls `matching.iou_distance`; `matching.py:101` returns ordinary IoU cost. The underlying `utils/metrics.py:bbox_ioa` is ordinary IoU, not a hidden L2 fusion. No upstream source was edited.

The paper's Section 3.2, Eq. 10 describes 0.25 times normalized L2 cost plus 0.75 times IoU cost; Eq. 11 then incorporates confidence. Therefore these results must be called **public-code BoT-SORT/GMC evaluation**, not reproduction of the full paper or its competition score. References: [official repository](https://github.com/earth-insights/Dist-Tracker), [paper](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.pdf).

## Protocol

- 14,201 consecutive frames, 100 FPS, 1920x1080 source; 14,199 reviewed frames and 7,982 GT boxes. Uncertain frames 9362 and 9363 excluded only from scoring, not from tracker updates.
- Existing Frozen-P3 + Add-on P2, PT FP32 input 960x544, detector conf=0.03, NMS IoU=0.45. No retraining, detector changes, additional low-confidence boxes or GT-guided tracker decisions.
- Source SHA256: `2629efe7da36dfd079985344653a7a4ecaa25f651954e5258a53c50e960d6b86`.
- Weights SHA256: `61e9a3669f964e59e96e9b2a24bf7705b945e184ed5a0628be33ee44d8b4bb59`.
- Detection-cache SHA256: `c1edc4cd5b13756389a33351d631066cc7c4bb9b1c19420ceae33233112ddc29`.
- Primary metrics use exact detector boxes selected by the tracker, IoU >= 0.5, so geometry smoothing cannot masquerade as association improvement. Upstream Kalman-filter output geometry is separately scored in every `summary.json`.
- No area/aspect-ratio filtering, ReID, interpolated or predicted boxes. Public-code activation rules are retained (normally confirmation on the next match, first-frame exception), not replaced by our native tracker's three-hit rule.
- Actual source FPS is passed to BOTSORT: `track_buffer=30` becomes 100 frames / one second. This differs from the repository's generic video callback hard-coding 30 FPS, deliberately matching the existing native one-second buffer.
- Upstream `GMC(method="sparseOptFlow", downscale=2)` is run causally on original consecutive images. Matrices are cached once for identical replay across configurations; the sparse-flow implementation does not use the detector boxes. Its built-in insufficient-feature identity fallback remains unchanged.

## Results

"Continuous ID changes" counts ID changes between correct observations within an uninterrupted positive-GT segment. "Adjacent changes" requires adjacent correctly tracked frames. These are single-GT diagnostics, **not standard MOT IDSW or IDF1**. Fewer changes can also conceal incorrect merging; cross-video/multiple-target validation remains necessary.

| Method | TP | FP | Precision | Recall | F1 | Continuous ID changes | Adjacent changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Detector only | 5302 | 3806 | 58.21% | 66.42% | 62.05% | n/a | n/a |
| Original delivered RK tracker | 5086 | 2677 | 65.52% | 63.72% | 64.60% | 153 | 35 |
| Prior RK active-first, match=0.92 | 5091 | 2686 | 65.46% | 63.78% | 64.61% | 99 | 12 |
| Prior RK active-first, match=0.96 | 5189 | 2872 | 64.37% | 65.01% | 64.69% | 49 | 9 |
| Public default thresholds, no GMC | 3758 | 1312 | 74.12% | 47.08% | 57.59% | 347 | 65 |
| Public low thresholds, score fusion, no GMC | 3669 | 1251 | 74.57% | 45.97% | 56.87% | 352 | 66 |
| Public low thresholds, no score fusion, no GMC | 4451 | 2018 | 68.81% | 55.76% | 61.60% | 413 | 107 |
| Public default thresholds + GMC | 4889 | 2278 | 68.22% | 61.25% | 64.55% | 29 | 2 |
| Public low thresholds, score fusion + GMC | 4688 | 2037 | 69.71% | 58.73% | 63.75% | 38 | 0 |
| **Public low thresholds, no score fusion + GMC** | **5287** | **3016** | **63.68%** | **66.24%** | **64.93%** | **6** | **2** |

The final row improves recall by 2.52 percentage points but loses 1.84 precision points versus the original delivered tracker: +201 TP and +339 FP. Correct detector observations suppressed by tracking fall from 216 to 15. Using its upstream Kalman boxes instead gives TP=5272, FP=3031, P=63.50%, R=66.05%, F1=64.75%.

The direct on/off GMC comparison is particularly important: with the same adapted thresholds and score fusion disabled, continuous ID changes fall from 413 to 6 and recall rises from 55.76% to 66.24%. This supports camera-motion compensation as the major useful component here; it does not establish that an unavailable L2 implementation caused the gain.

Remaining six continuous-segment ID changes occur at frames 1025, 1028, 5778, 6681, 9013 and 9665 (10.25, 10.28, 57.78, 66.81, 90.13 and 96.65 seconds). The first two are adjacent-frame switches. Tracking is improved, not perfect.

## IDs and Configuration

Original RK tracking allocated 582 IDs; 160 appeared as confirmed tracks and 422 never confirmed. The selected public-code candidate allocated 297 IDs, of which 56 appeared as confirmed tracks. These counters include false tracks and tentative births; they are not counts of real UAVs or ID switches. **No ID remapping, counter reset or forced single-object ID was used.**

Selected candidate parameters:

```yaml
tracker_type: botsort
track_high_thresh: 0.03
track_low_thresh: 0.01
new_track_thresh: 0.10
track_buffer: 30
match_thresh: 0.8
fuse_score: false
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: false
```

The immutable input cache contains only scores >= 0.03, so the low-confidence second stage has no additional boxes in this experiment. With score fusion enabled, the cost is `1 - IoU * score`; for a score of 0.10, even perfect overlap has cost 0.90 and cannot pass match=0.80. This is why merely reducing the input threshold is not sufficient.

## Artifacts and Reproduction

- Local upstream: `/Users/czyczyyzc/Documents/codes/Dist-Tracker`.
- Server upstream: `/mnt/chenziye/codes/Dist-Tracker`, cloned from the exact local Git bundle.
- Adapter locally: `/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/scripts/anti_uav/evaluate_dist_tracker_cache.py`.
- Adapter on server: `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/evaluate_dist_tracker_cache.py`.
- Final server results: `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/Video00009_dist_tracker_final_20260917/`.
- Original GMC cache on server: `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/Video00009_dist_tracker_20260917/gmc.npz`, with adjacent `gmc_protocol.json`.
- Final local results: `final/comparison.json`, `final/<configuration>/summary.json` and `tracks.jsonl`. `allocated_ids` is the real upstream counter; `observed_candidate_ids` excludes candidates immediately removed by upstream duplicate suppression.
- Original replay and GMC cache locally: `full/`. A second replay with the corrected ID-counter reporting produced byte-identical `tracks.jsonl` for all six variants.
- Local video: `visualization/Video00009_RK_vs_Dist_public_GMC_10x_slow.mp4` (16 seconds, H264, 1920x720, 160 frames at 10 FPS, 10x slow).

The video uses the same four source intervals as the previous diagnostic montage: frames [950,990), [2190,2230), [8985,9025), [11940,11980). Columns are detector, original RK confirmed tracks, selected public-code confirmed tracks, and detections plus confirmed IDs. Yellow pending detections in the fourth column are a visualization aid, **not recovered tracks and not included in tracker metrics**. Clean-source crops are enlarged before drawing one-pixel corner boxes; GT is never drawn and is used only to center offline diagnostic crops. The earlier `full/` replay is the montage source and is byte-identical to `final/` tracking outputs.

Run the adapter with `--help` for the input paths. It checks video, weights, GT and detection hashes, isolates the upstream Ultralytics import, advances empty frames, and rejects fabricated/stale observations. `--skip-gmc` is the no-GMC control; `--gmc-cache <original-results-directory>` reuses verified matrices. Use a new output directory for each run. CPU replay requires NumPy, SciPy, OpenCV, Torch, PyYAML and lap; no GPU inference is performed. The server uses `.venv/bin/python` and lap from the isolated `.codex_work/tracktrack_deps` directory.

Server CPU association took about 3.22 seconds for 14,201 frames; original-resolution GMC took 224.66 seconds. Adding the separately timed components gives approximately 62.32 FPS, excluding decoding, detector inference, rendering and IO. **This is not RK3588/RK3576 FPS**; GMC needs a separate native/RGA/downscaled implementation and real board measurement before deployment.

## Verification and Limits

All 21 related Python tracker tests pass on the server. Locally 20 pass and one existing renderer test is blocked by the system Python missing `psutil`; the new adapter's three tests pass locally and on the server. Video metadata, selected original-frame panels and a frame decoded from the encoded MP4 were checked. Final result archive SHA256 matches between server and local: `89bcd9628206e3b7007d97ee8594b76ea4d9f2c164aa332f00438b0a4a05f52a`.

Code was synchronized using local Git push followed by server Git pull from a bundle. No model, board binary, startup service or deployment preset was replaced. The previously started TrackTrack experiment and native continuity-policy work were paused, not enabled by this experiment. Video00004 and multi-object/reappearance scenes still need independent checks before choosing a deployment replacement.

## Full-Length Visualization

The full video is `full_visualization/Video00009_Dist_public_GMC_conf003_full.mp4`, under the local directory `/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/Video00009_dist_tracker_20260917/`. It contains all 14,201 consecutive source frames at the original 100 FPS: 142.01 seconds, H264, 1600x784, no audio. The playback rate is not a hardware inference-speed measurement.

This is a rendering of the exact `final/conf003_unfused_gmc/tracks.jsonl` result, not another inference or parameter-selection run. Only `displayed_tracks` (confirmed, currently observed tracks) are drawn using the associated detector boxes and original IDs. There is no GT, pending-detection overlay, predicted/interpolated box, ID remapping, area filter or skipped empty frame. The two right-hand insets follow the highest-score observed tracks. They are cropped from the clean source and resized before drawing one-pixel corners, with adaptive crop size to retain large targets. No GT is read, including for crop positioning.

`scripts/anti_uav/render_cached_tracker_result_video.py` verifies source/model/detection provenance, frame counts and timestamps, unique track/observation assignments, and each displayed box against its associated original detection. `full_visualization/protocol.json` and `summary.json` record input/output hashes and rendering metadata. This full-video output count includes the two uncertain frames omitted only from evaluation, so it is not necessarily identical to the reviewed-subset count in the comparison table.

Reproduce locally with a fresh output directory:

```bash
python3 scripts/anti_uav/render_cached_tracker_result_video.py \
  --source .codex_work/video00009_tracker_diagnosis/Video00009_original.mp4 \
  --detector-dir deliverables/Video00009_expanded28_20260917 \
  --tracker-dir deliverables/Video00009_dist_tracker_20260917/final/conf003_unfused_gmc \
  --output deliverables/Video00009_dist_tracker_20260917/full_visualization
```

Rendering needs only Python, NumPy, OpenCV and ffmpeg/ffprobe; it does not import YOLO, load model weights, use a GPU, or require the upstream tracker runtime. The dedicated renderer tests can run independently with `python3 -m unittest discover -s tests -p test_render_cached_tracker_result_video.py -v`.
