# YOLOv8 Anti-UAV Alerting Demo

This example implements a safe, alerting-only anti-UAV perception loop:

- `detection`: YOLO full-frame and optional tiled detection for tiny targets
- `tracking`: pluggable single-target tracker backends through `solutions.available_trackers()`
- `recovery`: ROI re-detection and full-frame fallback after tracker drift or loss
- `review`: human confirmation before a target becomes an active alert
- `recording`: JSONL state logs, alert event logs and optional alert crops

It does not expose any control or actuation interface.

## Run

```bash
python examples/YOLOv8-Anti-UAV-Perception/anti_uav_perception.py \
  --model yolov8n.pt \
  --source /path/to/video.mp4 \
  --target-classes drone,uav \
  --tracker template_match \
  --detect-interval 8 \
  --tile-size 640 \
  --state-log runs/anti_uav/states.jsonl \
  --alert-log runs/anti_uav/alerts.jsonl \
  --alert-crops runs/anti_uav/crops \
  --save-video runs/anti_uav/demo.mp4 \
  --show
```

## Manual confirmation

When `--show` is enabled:

- Press `c` to confirm the current target and raise an alert.
- Press `r` to reject the current target.
- Press `q` to quit.

Use `--no-manual-confirmation` to auto-confirm alerts after a short warmup when you are running unattended offline replays.

## IR and tiny-target modes

- `--input-mode ir --clahe` adapts the detector input for infrared-style footage.
- `--tile-size 640` enables tiled detection, useful when the target occupies very few pixels.
- ROI re-detection is enabled by default and can be disabled with `--disable-roi-redetect`.

## Outputs

`state-log` records one frame-level status per line, including `status`, `confirmation_state`, `alert_active`, and `detector_mode`.

`alert-log` records alert lifecycle events such as:

- `alert_raised`
- `alert_cleared`
- `alert_rejected`

`alert-crops` saves image crops for emitted alerts so you can review false positives and misses later.
