# Native RKNN + Dist + GMC

Production inference uses only the `anti_uav_dist_native` C++ executable and
native shared libraries. Python scripts in the parent directory are **offline
verification tools**, not runtime dependencies or subprocesses of this executable.

## Build and run on the CM5

Requires Linux/aarch64, a working RKNN runtime/driver, g++, OpenCV development
libraries (`pkg-config opencv4`), and OpenSSL development libraries. The existing
detector shared library is reused unchanged to isolate tracker/pipeline changes.

```bash
cd /home/orangepi/ultralytics_yolov8_ziye
bash scripts/anti_uav/dist_native/build.sh \
  /home/orangepi/deployments/expanded28_dist_native_20260920
bash scripts/anti_uav/dist_native/run_on_board.sh \
  --output /home/orangepi/deployments/expanded28_dist_native_20260920/full_native \
  --save-observations
```

Output directories must be new. The launcher uses `exec`, not Python. The
executable loads the detector, tracker and GMC libraries directly with `dlopen`.
Use `--frames 2000` for a short test; `--warmup 100` is the default. `--frames 1
--warmup 0` measures first-result latency in a fresh process. `--detector-only`
omits tracker and GMC. `--no-pyramid-cache` is the GMC control experiment.

Deployment assets:

- RKNN: `/home/orangepi/deployments/expanded28_dist_20260920/detector_960x544_int8.rknn`
- Video: `/home/orangepi/deployments/expanded28_dist_20260920/Video00009_original.mp4`
- Detector library: `/home/orangepi/deployments/expanded28_dist_opt_20260920/libanti_uav_detector.so`
- New executable/libraries: `/home/orangepi/deployments/expanded28_dist_native_20260920/`

The launcher accepts `ANTI_UAV_NATIVE_DEPLOY`, `ANTI_UAV_BASE_DEPLOY`, and
`ANTI_UAV_DETECTOR_DEPLOY` overrides. Alternatively invoke the executable with
explicit `--model`, `--video`, `--detector-library`, `--tracker-library`,
`--gmc-library`, and `--output` arguments.

## Pipeline and fixed comparison settings

CPU OpenCV video decode -> ready-worker dispatch to three independent RKNN
contexts (one NPU core each) -> ordered native Dist association. Native GMC runs
in a separate ordered thread alongside detector inference. The queue is bounded
to nine in-flight frames. No frame skipping, cached detections, Python, GT,
rendering, or video encoding is used in inference. This is not an MPP/RGA decode
implementation. Every output box comes from that frame's detector result.

- Model: unchanged expanded-28 Frozen-P3 + Add-on P2, 960x544 INT8.
- Model SHA256: `2afa76f9f093265b0e347d26285ed15c98c449d958eed040f27e7345f29deb53`.
- Detector: conf 0.03, NMS IoU 0.45, maximum 100 boxes; no area cutoff.
- Tracker: high 0.03, low 0.01, new 0.10, match 0.8, buffer 30 scaled by source FPS;
  no ReID or score fusion. These tracker settings are fixed in this comparison.
- GMC: resize BGR to 320px width then gray, at most 128 corners, refresh every
  five frames or when support is insufficient, LK and robust affine estimation
  on every frame. Cached LK pyramids are reused without skipping GMC updates.
- CPU set 4,5,6,7; detector workers pinned to 4,5,6; OpenCV threads 1.

`summary.json` records throughput excluding warmup, stage timings, first-result
latency, model/library hashes, per-core frame counts and thermal/frequency
samples. Stage times overlap and must not be added to infer pipeline FPS.
Process-entry time excludes the OS loader; use the offline
`benchmark_dist_startup.py` harness for process-launch latency. Neither is
power-on or camera-to-display latency. No thermal protections are changed.

## Behavior verification

`tracker.cpp` ports the actual public tracker code executed by this deployment,
including XYWH Kalman state, three matching stages, duplicate removal and lost
track lifecycle. It is not the older lightweight RK-BoT-SORT renamed as Dist.
The public implementation does not implement every component described in the
Dist paper; do not claim unimplemented paper features.
The executed upstream is https://github.com/earth-insights/Dist-Tracker at
`396c359e1aa8be4fd5e81a02626cb1ee3867cf7c`.

- `verify_native_dist.py`: synthetic threshold/duplicate/expiry cases and complete
  14,201-frame cached replay; checks IDs, current detector indices and boxes.
- `verify_native_gmc.py`: Python/reference versus C++ raw/cached image processing.
- `compare_native_dist_observations.py`: strict full native pipeline comparison;
  use `--prefix 2000` only for an explicitly limited short run.
- `evaluate_dist_optimized.py`: offline reviewed-GT scoring. GT never enters the
  native runtime.

Replay/ctypes validation time is not pure C++ kernel time or live pipeline FPS.
Do not substitute cached-replay numbers for full video measurements. On the
heatsink-only board, temperature can materially change sustained throughput.

## Licenses

The tracker port retains the public Ultralytics/Dist tracker AGPL-3.0 licensing
notice. The vendored `lap` v0.5.12 solver retains its BSD-2-Clause license under
`third_party/lap/LICENSE`. Review applicable licenses before redistribution.
The solver sources are unmodified from https://github.com/gatagat/lap v0.5.12,
commit `600c210d9bef793ee0fe502cbc350e676a6e083a`. Finite cost-limit padding
matches its Python wrapper; do not substitute a different assignment solver
without rerunning equivalence checks.

## Validated Software Input Optimization

The original launcher/defaults remain `--decoder opencv --preprocess opencv`.
A separate RK3588S-tested launcher selects four frame-decoding threads and
fused preprocessing, without replacing the original deployment:

```bash
bash scripts/anti_uav/dist_native/run_optimized_on_board.sh \
  --output /home/orangepi/deployments/expanded28_dist_latency_20260920/new_run \
  --save-observations
```

Its executable and all three libraries are in
`/home/orangepi/deployments/expanded28_dist_latency_20260920/`; override that
directory with `ANTI_UAV_LATENCY_DEPLOY`. Build the executable with `build.sh`
and rebuild the detector with `scripts/anti_uav/build_dist_detector_on_board.sh`.
FFmpeg development libraries are required for the direct software decoder.

- `--decoder ffmpeg --decode-threads 4 --decode-threading frame`: native FFmpeg
  software H264/HEVC decoding with explicit frame threading. The bare FFmpeg
  option defaults to one thread/slice mode, which was slower in this test.
  No frame discard, low-delay flag or changed presentation order is requested.
  CPU BGR conversion is still performed. Frame threading can buffer future
  compressed frames; file-read timings are not live-camera capture latency.
- `--preprocess fused`: one-pass half-size BGR-to-RGB downsampling with ARM NEON,
  cached letterbox padding and the existing contiguous RGB copy to RKNN memory.
  Other resize ratios fall back to the existing OpenCV resize/color conversion.
  This needs a rebuilt detector library exposing `au_detector_fused_supported`;
  the existing released detector library will deliberately fail this check.
- `--decoder rkmpp` / `--preprocess rga`: experimental hardware paths. The first
  combined MPP trial crashed; do not select it as the deployment default.

The complete 14,201-frame Video00009 optimized run exactly matched reference
detector boxes/scores, displayed tracks/IDs and GMC warps. Board OpenCV 4.5.4
pixel tests also passed for 264 randomized color/grayscale/stride layouts.
See `deliverables/expanded28_dist_latency_20260920/README.md` for measurements
and first-frame limitations. This remains a software decode path, not zero-copy.
Do not generalize file benchmark latency to compressed live streams or camera
sensor-to-result latency. Other resize ratios use the OpenCV fallback.
