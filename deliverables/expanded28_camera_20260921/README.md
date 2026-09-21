# Live Camera Latency: RK3588S CM5, Vendor 5.10.198

## Verified Result

Measured on 2026-09-21 using the new board's live camera, not decoded video or
cached frames. Native path: V4L2 raw8 capture -> fused grayscale resize/RGB
replication -> separate RKNN worker contexts, concurrent C++ GMC -> ordered
C++ Dist association. Inference contains no Python. Python only audits CSV/JSON
after the experiment. The model is the same expanded-28 Frozen-P3 + Add-on P2
960x544 INT8 model used in the previous 85.06 FPS video benchmark.

| Run | Output frames | Output FPS | Read-to-result mean | Driver-timestamp-to-result mean / p95 |
| --- | ---: | ---: | ---: | ---: |
| One NPU worker, latest, 4 capture buffers | 2,000 | 33.49 | 29.79 ms | 41.28 / 44.99 ms |
| Three NPU workers, latest, 4 buffers | 2,000 | 86.05 | 34.73 ms | 44.18 / 49.86 ms |
| Three NPU workers, latest, 4 buffers | 12,000 | 86.39 | 34.60 ms | 44.07 / 49.66 ms |
| Three NPU workers, FIFO, 4 buffers | 2,000 | 89.67 | 33.63 ms | 61.32 / 72.49 ms |
| Three NPU workers, latest, 3 buffers | 2,000 | 84.75 | 35.08 ms | 43.70 / 49.57 ms |
| Three NPU workers, latest, 2 buffers | 1,000 | 80.03 | 37.43 ms | 40.79 / 46.40 ms |

The selected default remains three workers, three in-flight jobs, latest-frame
capture and four capture buffers. Two buffers is a short-test lower-latency
alternative with reduced throughput, not a long-run validated improvement.
FIFO used nine in-flight slots, matching the previous video setting; the
low-latency runs use one slot per worker. This is a configuration comparison,
not a one-variable ablation.

The 12,000-output run lasted approximately 139 seconds. The final 2,000 outputs
ran at 86.50 FPS, with peak sampled temperature 44.384 C. This is a roughly
2.3-minute test, not an indefinite thermal guarantee. FPS and steady stage
statistics exclude the first 100 processed frames; all frames still produce
results. `summary.json` also retains all-frame FPS and startup timings.

The camera's sequence/timestamps imply approximately 120 FPS. The long run
skipped 4,667 source sequence positions between first and last processed frames
(about 28.0%); 3,997 were explicitly drained by latest-frame selection. The
remaining gaps were not delivered by the driver to this consumer. No claim of
processing every 120 FPS frame is made. One-worker skipping was 72.1%.

## Timing Boundaries and Startup

Three fresh-process trials per condition, median values:

| Startup metric | No NPU warmup | Three warmup calls per worker |
| --- | ---: | ---: |
| First camera read / STREAMON to first raw frame | 109.48 ms | 110.22 ms |
| Start reading / STREAMON to first result | 139.74 ms | 139.99 ms |
| First driver frame timestamp to result | 38.65 ms | 38.78 ms |
| Process entry to first result | 206.32 ms | 477.68 ms |
| Explicit NPU warmup cost | 0 ms | 270.56 ms |

The process clock starts inside `main`'s translation unit; it does not include
the shell or dynamic loader. OS caches were not cleared. These are fresh-process
tests, not board power-on tests. Warmup calls execute before camera STREAMON and
are separate from the 100-frame statistics exclusion used in sustained runs.

Removing HEVC decode does not remove sensor/driver STREAMON startup. The old
video first-read-to-result median of 82.54 ms therefore cannot be replaced by
the live steady value of 34.60 ms. Keep the camera and model running for low
steady latency; this experiment found no first-frame benefit from explicit
NPU warmup.

All audited rows have monotonic V4L2 timestamps, flags 8193 (timestamp source
reported as EOF). The driver timestamp-to-result metric includes the age of
the frame before application dequeue, unlike read-to-result. It is NOT a
verified exposure-to-result or photon-to-display measurement; display, exposure
start and physical sensor timestamp accuracy were not measured. The driver
chooses the timestamp source, as described in the
[Linux V4L2 buffer documentation](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/buffer.html).

The 12,000-frame mean stage timings were:

| Stage | Mean |
| --- | ---: |
| Camera read, wait and copy (legacy JSON key `decode`) | 1.751 ms |
| Raw frame copy alone | 0.473 ms |
| Fused preprocess | 0.680 ms |
| RKNN inference per frame | 31.597 ms |
| Detection postprocess | 0.303 ms |
| GMC, concurrent with detector | 3.116 ms |
| Dist association | 0.005 ms |
| Driver timestamp to application dequeue | 10.692 ms |

These stages overlap; do not sum all columns as a serial pipeline. Multi-core
throughput is not reciprocal per-frame latency. One-worker inference averaged
28.534 ms versus three-worker 31.705 ms in the matched 2,000-frame runs, consistent
with shared-resource contention. Read wait and output ordering also differ.

## Scope and Limitations

- Model SHA256: `2afa76f9f093265b0e347d26285ed15c98c449d958eed040f27e7345f29deb53`.
- Conf 0.03, NMS IoU 0.45, same Dist thresholds; GMC width 320, 128 corners,
  refresh interval 5, cached pyramids. Each NPU context has separate input/output buffers.
- Kernel 5.10.198, RKNN runtime 2.3.2, NPU driver 0.9.8, OpenCV 4.5.4.
- NPU 1 GHz, big CPUs 2.304 GHz, DDR 1.56 GHz; existing performance governors
  retained. No overclocking, thermal disablement or persistent service changes.
- Camera `/dev/video11`: 1920x1080, BA81 raw8, 2048-byte input stride.
  Raw8 is explicitly treated as monochrome and replicated into RGB, not demosaiced.
- Live scene is an upside-down indoor view, partially occluded, without an
  annotated UAV trajectory. The long run produced 84 detections and zero
  displayed tracks. Dist/GMC code executes, but this does NOT validate UAV
  tracking accuracy or association latency under many active tracks.
- Tracker timing remains update-step based, as in the video implementation.
  Skipped live source frames are logged but do not trigger extra Kalman updates;
  variable-frame-interval tracking accuracy needs separate validation.
- No display/encoding is included. A first raw frame is saved only after timing
  completes. CSV timing logs are written during the run.
- The earlier video result used another board, kernel 6.1.99, DDR 2.112 GHz,
  different temperatures and an outdoor video. Same model does not make these
  runs a controlled camera-vs-decoder A/B experiment.

## Deployment and Reproduction

Board code checkout (the older `_ziye` directory is an unpacked release, not Git):
`/home/orangepi/ultralytics_yolov8_ziye_code`

Board runtime/model/results:
`/home/orangepi/deployments/expanded28_camera_20260921`

Model: `/home/orangepi/deployments/expanded28_camera_20260921/detector_960x544_int8.rknn`

```sh
cd /home/orangepi/ultralytics_yolov8_ziye_code
bash scripts/anti_uav/dist_native/run_camera_on_board.sh \
  --frames 12000 \
  --output /home/orangepi/deployments/expanded28_camera_20260921/new_run
```

Choose a new output directory. For one worker append `--workers 1 --inflight 1`.
For startup tests use `--frames 1 --warmup 0`; explicit warmup is `--npu-warmup 3`.
`--camera-fps 120` declares the tracker rate; it does not program the sensor.

The measured binary was built from commit `030d07b752ae83f3f8fdbc6fb95adc2b77e8081f`.
Later reporting-only commits do not change the executable. Its SHA256 is
`1a3ccd4b2b3ce98c1f73dd5efa3a128f8f439e7a5b2453a623a5e53222fef9bc`.
Detector library SHA256:
`a04e31573f42301e59e329cc81d991dc1306eed2d9d245681edcfadc721e8e7a`.

`build_camera.sh` builds the matching executable and detector library together.
OpenCV development/runtime packages were extracted under `deps`, not installed
over system libraries. The original RKNN library under
`/home/orangepi/ultralytics_yolov8_ziye/lib` is used. Rebuild with:

```sh
D=/home/orangepi/deployments/expanded28_camera_20260921
export RKNN_INCLUDE=/home/orangepi/ultralytics_yolov8_ziye/src
export RKNN_LIB=/home/orangepi/ultralytics_yolov8_ziye/lib
bash scripts/anti_uav/dist_native/build_camera.sh "$D/bin" "$D/deps"
```

The board has no default Internet route. Git clone/pull used a temporary,
localhost-only Git relay over SSH from the Mac, after local GitHub push. The
47 server used `git pull --ff-only` from a bundle of the same pushed commits.
The board relay URL only works while that tunnel is active; it is not a public
repository endpoint. No unrelated local working-tree changes were committed.

## Evidence and Checks

Local evidence: this directory. Each run contains `summary.json`, `latency.csv`
and `first_raw_gray.png`; `comparison.json` is generated by:

```sh
python3 scripts/anti_uav/summarize_camera_benchmark.py \
  deliverables/expanded28_camera_20260921
```

The auditor checks frame counts, ordering, source sequence gaps, monotonic
timestamp validity, latency-summary agreement and identical model hashes.
All downloaded runs passed. Fused preprocessing tests passed 132 RGB layouts,
132 grayscale layouts and 10 invalid-input cases on both Mac ARM with
ASan/UBSan and the board's ARM NEON build. This checks pixel equivalence to RGB
replication, not detector accuracy against ground truth.
