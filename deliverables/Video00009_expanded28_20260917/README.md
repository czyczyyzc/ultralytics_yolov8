# Video00009: Expanded-28 Detector Visualization

Model: 28-video Frozen-P3 + Add-on P2, PT FP32 detector only.
Input 960x544, confidence 0.03, NMS IoU 0.45, max detections 100, GPU 6.
No ground truth is read or displayed; no tracker, track IDs or crosshairs are used.

## Video

`Video00009_expanded28_frozenP3_addonP2_conf003.mp4` is the full source clip:
14,201 frames, 100 FPS playback, 142.01 seconds (about 2 minutes 22 seconds).
It is not the 1,421-frame sampled validation subset and does not reuse its metrics.
Playback FPS is inherited from the source and is not a measurement of RK3588 inference FPS.

The output is H.264/yuv420p with faststart, 1600x784, and no audio. The full source
image is displayed at 1280x720 without cropping or aspect-ratio distortion.
One-display-pixel cyan corners indicate predictions. Two side panels show the
highest-confidence prediction locations, cropped from the clean original frame,
enlarged before drawing the boxes. Rank is confidence order, not track identity.
No previous-frame box is carried into a missed frame.

The original capture is 1920x1080 HEVC, 100 FPS. Source SHA256:

```text
2629efe7da36dfd079985344653a7a4ecaa25f651954e5258a53c50e960d6b86
```

Weights SHA256:

```text
61e9a3669f964e59e96e9b2a24bf7705b945e184ed5a0628be33ee44d8b4bb59
```

## Server Artifacts

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/Video00009_expanded28_visual_full_20260917
```

Source video:

```text
/mnt/andrew/video-labeler/videos/20260805_multirotor_sunny_frontlight_stationary_video00009_90af30d9.mp4
```

Model:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_addon/p2/weights/best.pt
```

Reproduction script: `scripts/anti_uav/render_pt_detector_video.py`.
`protocol.json` records input/model hashes and render settings. `summary.json`
records final frame count, duration, prediction totals and output SHA256.
`predictions.jsonl` records zero-based source-frame indices and original-pixel
xyxy coordinates plus confidence, including empty prediction lists.

## Checks

The 128-frame smoke render encoded and probed successfully. Panel checks covered
zero/two predictions, edge crops, unchanged source pixels and output dimensions.
The renderer verifies encoded frame count and duration before reporting completion.
Preview overlays are separate from model input; no synthetic replacement images
were used in this model's training.

The complete render finished successfully: 14,201 encoded frames, 142.01 seconds,
9,109 predictions across 8,468 frames. These are prediction counts, not TP/FP
metrics; the renderer does not read annotations. Output SHA256:

```text
26cf7025bebc3e16e88f72491efa1ab9e61531d5388a9f587d361f1f60b41b79
```
