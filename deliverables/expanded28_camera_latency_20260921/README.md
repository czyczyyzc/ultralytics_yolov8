# Live Camera Latency Optimization

Measured 2026-09-21 on the same RK3588S CM5 / vendor 5.10.198 as the preceding
camera benchmark. Real 1920x1080 raw8 camera, 960x544 INT8 Frozen-P3 + Add-on P2,
native C++ Dist + concurrent GMC. No cached detections, Python inference,
resolution/threshold changes, overclocking or thermal-policy changes.

## Results

All rows include capture, detector, ordered Dist and GMC. FPS and steady timing
exclude the first 100 outputs. Skipped source frames are counted from driver
sequence gaps; the sensor produces approximately 120 FPS.

| Configuration | Outputs | FPS | Read-to-result mean | Driver-to-result mean / P95 | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original lazy start, latest, 4 buffers, 3 workers | 6,000 | 86.34 | 34.62 ms | 43.98 / 49.75 ms | 28.0% |
| Overlapped start, latest, 4 buffers, 3 workers | 12,000 | 86.58 | 34.53 ms | 43.96 / 49.59 ms | 27.8% |
| Overlapped start, latest, 2 buffers, 3 workers | 6,000 | 79.95 | 37.46 ms | 41.08 / 47.46 ms | 33.4% |
| Overlapped start, fresh, 4 buffers, 1 all-core context | 2,000 | 30.00 | 33.27 ms | 36.62 / 38.91 ms | 75.0% |

The selected fast-start launcher keeps the original 3-worker / latest / 4-buffer
steady configuration. Its long-run latency and throughput are effectively the
same as baseline; the verified benefit is startup. Two buffers provides a modest
steady frame-age reduction at a throughput cost. Waiting for fresh frames and
using one context produces the lowest tested frame age, but only 30 FPS. It is
not a free speed improvement and is not the recommended default.

Additional 600-output screening runs, not equally long validation:

| Setting | FPS | Driver-to-result mean / P95 |
| --- | ---: | ---: |
| 3 workers, fresh, 4 buffers | 72.28 | 40.60 / 45.36 ms |
| 3 workers, fresh, 2 buffers | 62.45 | 39.36 / 41.84 ms |
| 1 worker, mask 0_1_2, latest | 34.31 | 40.57 / 44.33 ms |
| 2 workers, masks 0_1 and 2, latest | 64.00 | 40.68 / 46.03 ms |
| 2 workers, masks 0_1 and 2, fresh | 57.58 | 38.90 / 41.08 ms |

For this RKNN artifact, requesting cooperative core masks did not outperform
three independent contexts. An accepted mask is not proof of a proportional
multi-core speedup. No claim about other compiled models is made.

## Startup

Five alternating fresh-process trials per mode, no NPU prewarm, median:

| Measurement | Lazy start | Overlapped start |
| --- | ---: | ---: |
| Program entry to first result | 207.37 ms | 141.37 ms |
| STREAMON invocation to first result | 140.33 ms | 137.93 ms |
| Blocking STREAMON call | 103.44 ms | 102.23 ms |
| First read call to first result | 140.33 ms | 36.20 ms |
| First driver frame timestamp to result | 38.74 ms | 38.29 ms |

Program-entry latency improved by 31.8%, approximately 66 ms. Sensor STREAMON
runs in a separate initialization task while RKNN contexts/libraries load. The
first read now occurs after STREAMON returns, so its much smaller number must
NOT be presented as a 74% improvement to physical camera startup. STREAMON itself
still costs roughly 102 ms. A separate long-run first result was 154.47 ms from
program entry, illustrating startup variation beyond the five-trial median.

Program-entry excludes the shell and dynamic loader; caches were not flushed.
These are process restarts, not board power-on measurements.

## Reproduce

Code: `/home/orangepi/ultralytics_yolov8_ziye_code`

Deployment: `/home/orangepi/deployments/expanded28_camera_latency_20260921`

```sh
cd /home/orangepi/ultralytics_yolov8_ziye_code
bash scripts/anti_uav/dist_native/run_camera_fast_start_on_board.sh \
  --frames 12000 \
  --output /home/orangepi/deployments/expanded28_camera_latency_20260921/new_run
```

Append `--camera-buffers 2` for the measured approximately 80-FPS alternative.
Append `--workers 1 --inflight 1 --npu-masks 0_1_2 --camera-policy fresh` for
the measured approximately 30-FPS lower-frame-age alternative.
Append `--camera-start lazy` to disable overlap. Always use a new output folder.
The original `run_camera_on_board.sh` default deployment is unchanged.

`--camera-policy fresh` discards completed buffers before waiting for the next
completion. `--npu-masks` requires one mask per worker, and accepts explicit
0, 1, 2, 0_1 or 0_1_2 masks. Each context owns independent I/O buffers.
`--warmup` excludes initial outputs from statistics; it is not NPU prewarming.

Binary code commit: `9ded0bd`. Executable SHA256:
`aa9997747603e65a0e20b710f83e378ff1c76dc0448bcc2aa30baf7e8aaf2732`.
Model SHA256:
`2afa76f9f093265b0e347d26285ed15c98c449d958eed040f27e7345f29deb53`.
Detector, tracker and GMC libraries are byte-identical to the preceding camera
deployment. Runtime 2.3.2, driver 0.9.8, NPU 1 GHz, DDR 1.56 GHz retained.

The 12,000-output run took about 139 seconds, with peak sampled temperature
45.307 C. This is not an indefinite thermal-stability guarantee.

## Verification and Limits

All raw `summary.json`, `latency.csv` and first-frame images are retained locally
and in the board deployment. `comparison.json` is generated with
`scripts/anti_uav/summarize_camera_benchmark.py`; it audits every frame's index,
monotonic timestamp order, sequence gaps, model hash and summary/CSV agreement.
The new parser rejected six invalid option combinations. Board preprocessing
tests passed 132 RGB cases, 132 grayscale cases and 10 invalid-input cases.

Driver timestamp-to-result is NOT verified exposure-to-result latency and does
not include display. Driver flags are 8193 and timestamps use the monotonic clock.
The indoor scene is partly occluded and upside down, without a UAV trajectory.
The long run had zero detections/tracks. GMC executes, but active-target tracking
accuracy and heavily populated association latency are not validated here.
Capture skipping is explicit; the tracker still uses update-step timing.

Further latency reductions by removing P2 are a separate model comparison in
`../expanded28_p3_camera_20260921/`, not a same-model optimization.
