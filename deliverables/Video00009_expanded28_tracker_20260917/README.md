# Video00009: Expanded-28 Detector + RK-BoT-SORT

Model: 28-video Frozen-P3 + Add-on P2, PT FP32, input 960x544.
This visualization reuses every prediction from the preceding detector-only
video at confidence 0.03, NMS IoU 0.45, max detections 100. There is no new
detector inference, frame sampling, ground-truth input or synthetic image input.

## Output

`Video00009_expanded28_frozenP3_addonP2_RKBoTSORT_conf003.mp4`

Full source video: 14,201 frames, 142.01 seconds, 100 FPS playback, no audio.
Playback FPS is not RK3588/RK3576 inference throughput. Output is H.264,
yuv420p, faststart, 1600x784, with a complete 1280x720 main view and two
clean-source zoom crops. Cyan 1px corner boxes show confirmed observed tracks;
IDs appear beside boxes and beneath zoom crops. No GT, crosshairs, trails or
unobserved predicted boxes are displayed. Crop rank follows confidence, not ID.

## Tracker

The replay builds and calls the same native C++ tracker used by the board:
`scripts/anti_uav/rknn_yolov8_native/detector_based_tracker.hpp`, through
`tracker_c_api.cpp` and the existing `NativeTracker` bridge.

| Parameter | Value |
| --- | --- |
| High / low association confidence | 0.03 / 0.01 |
| New track confidence | 0.10 |
| First / second match cost | 0.92 / 0.92 |
| Lost track buffer | 1.0 seconds |
| Prediction display grace | 0.0 seconds |
| Confirmation hits | 3 |
| Source FPS | 100 |
| Pose input | None |

Important: the exact detector-only cache was truncated at confidence 0.03.
Scores in [0.01, 0.03) therefore cannot participate in second-stage association.
This is a strict same-detections comparison, not the separate conf=0.01 board
pipeline. A detection does not necessarily yield a visible confirmed track:
new-track confidence and confirmation gates intentionally suppress tentative
tracks. Track IDs and output counts are not ID-switch or precision metrics.

## Provenance

Detector model on server 47:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_addon/p2/weights/best.pt
```

Detector input cache:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/Video00009_expanded28_visual_full_20260917
```

Tracker output on server 47:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/Video00009_expanded28_tracker_full_20260917
```

Reproduction script: `scripts/anti_uav/render_cached_rk_botsort_video.py`.
`protocol.json` records detector settings, input hashes and native tracker source
hashes. `tracks.jsonl` records all raw tracker outputs and displayed outputs for
each zero-based source frame. `summary.json` records final encoded frame count,
duration, displayed-track totals and output SHA256.

## Validation

Four unit tests passed on server 47: cache parsing, invalid cache rejection,
confirmation/prediction filtering, and empty/edge-crop/source-preservation rendering.
A native C++ bridge check passed confirmation at three hits, omission of predicted
boxes on a missed frame, and same-ID reassociation on the following observation.
A 128-frame smoke render encoded and passed frame-count and duration checks.
The full renderer validates input video/cache provenance and final output frame
count and duration before marking the job complete.
