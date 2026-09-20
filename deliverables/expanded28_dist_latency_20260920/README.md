# RK3588S Decode and Preprocessing Validation

## Scope and Selected Configuration

Measured on the connected Orange Pi CM5, Ubuntu 22.04, kernel
6.1.99-rockchip-rk3588, RKNN 2.3.2 and OpenCV 4.5.4. Production inference is
entirely C++: software decode, three RKNN workers, ordered GMC and Dist.
Python is used only for offline validation and external startup timing.

The separate optimized launcher selects `--decoder ffmpeg --decode-threads 4
--decode-threading frame --preprocess fused`. The original launcher and stable
deployment are unchanged. No service, thermal protection or governor was changed
in this test round. MPP/RGA was not enabled or retried.

Fixed protocol: full Video00009, 1920x1080 HEVC 100 FPS, 14,201 frames, unchanged
expanded-28 Frozen-P3 + Add-on P2 RKNN INT8 960x544, conf 0.03, NMS 0.45,
three independent NPU contexts, in-flight limit 9, CPU set 4-7, GMC 320px/128
corners/refresh 5, unchanged Dist thresholds. No skipped frames, rendering,
encoding or cached inference. Observations were written during full tests.
FPS and steady stage statistics exclude the first 100 frames, but those frames
are processed and their outputs are not withheld. The first-frame trials use
one frame, zero warmup and a fresh process; filesystem caches are not cleared.

## Full-Video Results

| Configuration | FPS | Read/decode mean | Preprocess mean | Ordered latency mean / p95 |
| --- | ---: | ---: | ---: | ---: |
| Same-build OpenCV decode + OpenCV preprocessing | 79.60 | 7.04 ms | 3.52 ms | 49.46 / 64.19 ms |
| OpenCV decode + fused preprocessing | 84.73 | 7.14 ms | 1.08 ms | 46.49 / 60.98 ms |
| FFmpeg 4 frame threads + fused preprocessing | 85.06 | 7.42 ms | 1.08 ms | 45.07 / 58.38 ms |

The measured full-video gain is 6.87% throughput, 69.27% less preprocessing
time and 8.89% less mean ordered latency. Most benefit is from fused
preprocessing; average full-video decode time did NOT improve. The optimized
last 2,201 frames ran at 81.47 FPS. Do not present the 92.90 FPS short test as
the sustained full-video result.

Changing only preprocessing already delivered most of the gain. Four-thread
FFmpeg versus OpenCV with fused preprocessing differed by only 0.33 FPS in
these single full runs: no statistically significant decoder throughput win is
claimed. Keep OpenCV + fused as the lower-change alternative, especially when
process startup matters more than read-to-result latency.

The board has a heatsink but no fan. Sampled optimized-run temperatures peaked
at 78.54 C. Sampled big-core clocks stayed at 2.352 GHz, NPU at 1 GHz and DDR
at 2.112 GHz. Runs were sequential, not thermally controlled repeated trials;
temperature and scene complexity still affect performance. A complete video
is about 167 seconds of processing, not an indefinite thermal stability test.

## First-Frame Results

Three fresh-process trials per configuration, medians:

| Measurement | Same-build control | OpenCV + fused | FFmpeg 4 + fused |
| --- | ---: | ---: | ---: |
| First read/decode to BGR | 56.32 ms | 55.37 ms | 53.14 ms |
| First preprocessing | 4.22 ms | 1.46 ms | 1.44 ms |
| Start reading first frame to ordered result | 89.00 ms | 84.98 ms | 82.54 ms |
| Process entry to first result | 154.43 ms | 150.38 ms | 148.45 ms |
| External process launch to first result | 247.58 ms | 243.70 ms | 256.30 ms |

The read-to-result improvement is 6.46 ms, not a removal of the roughly 53 ms
first HEVC decode. External launch timing did not improve in these trials;
it includes the shell, dynamic loader, initialization and stdout observation.
Do not conflate it with process-entry timing or promise a cold-start benefit.
The first-frame GMC costs about 3.2 ms but overlaps preprocessing/inference;
it must not be added serially to the detector path. Independent column medians
are not the decomposition of a single trial; raw trial summaries are retained.

The input stream has no B frames, tiles or wavefront entropy sync. Empirically,
single/slice-thread decode did not remove the first-read cost. Four frame
threads improved throughput over one thread but can buffer future compressed
frames. All timings here start at application file read, not sensor exposure,
network arrival, power-on or display. Live camera latency needs a separate
timestamped capture test and may favor a different decoder configuration.

## Short Ablation and Correctness

2,000-frame runs, same warmup and protocol:

| Configuration | FPS | Mean preprocessing | Mean ordered latency |
| --- | ---: | ---: | ---: |
| OpenCV + OpenCV | 84.72 | 3.58 ms | 46.27 ms |
| OpenCV + fused | 91.57 | 1.12 ms | 42.90 ms |
| FFmpeg 1 slice thread + OpenCV | 52.43 | 3.25 ms | 50.97 ms |
| FFmpeg 1 slice thread + fused | 51.81 | 0.95 ms | 48.75 ms |
| FFmpeg 4 frame threads + fused | 92.90 | 1.12 ms | 42.13 ms |
| FFmpeg 8 frame threads + fused | 92.96 | 1.12 ms | 42.45 ms |

One decoder thread is rejected for throughput. Eight threads had no material
short-test advantage over four. These are single-run observations, not
confidence intervals.

- Full optimized output exactly matches the stable reference for all 14,201
  frames: detector boxes/scores, displayed track IDs/current boxes and GMC
  warps, with 8,169 displayed observations. See `full_equivalence.json`.
- Same-build full control also matches: `control_equivalence.json`.
- OpenCV + fused full output also matches: `opencv_fused_equivalence.json`.
- Board ARM NEON preprocessing matched OpenCV 4.5.4 exactly on 264 randomized
  RGB/grayscale/strided layouts and guard regions: `board_pixel_equivalence.json`.
- Mac ARM sanitizer tests passed 132 layouts plus five invalid-input cases;
  server scalar tests matched OpenCV 4.11.0 on 264 cases.

Fused preprocessing performs exact 2x reduction plus BGR-to-RGB in one NEON
pass and reuses letterbox padding. Other ratios use the existing OpenCV
fallback. The final copy to RKNN input memory remains: this is NOT zero-copy.
The previous hardware attempt logged RGA I/O errors/software fallback and
crashed. It is not a supported optimized path in this release.

## Deployment and Reproduction

Board code: `/home/orangepi/ultralytics_yolov8_ziye`

Board binaries/results: `/home/orangepi/deployments/expanded28_dist_latency_20260920`

Server code: `/mnt/chenziye/codes/ultralytics_yolov8`

Model: `/home/orangepi/deployments/expanded28_dist_20260920/detector_960x544_int8.rknn`

Model SHA256: `2afa76f9f093265b0e347d26285ed15c98c449d958eed040f27e7345f29deb53`

Executable SHA256: `f14efb8d55bd6ee441ed8bb66d285dff9873896df087229d0bcc030418747941`

Detector library SHA256: `76ad8d81b150ee44ba93919745101e6cb195361bc6bc4497179299b4ecaa3c28`

The executable was built from commit `4a1e18e`; its launcher was added in
`61e411a`. Documentation-only changes do not require a binary rebuild.

```bash
cd /home/orangepi/ultralytics_yolov8_ziye
bash scripts/anti_uav/dist_native/run_optimized_on_board.sh \
  --output /home/orangepi/deployments/expanded28_dist_latency_20260920/new_full_run \
  --save-observations
```

Use a fresh output directory. For the control append
`--decoder opencv --preprocess opencv`; to isolate preprocessing use
`--decoder opencv --preprocess fused`. Add `--frames 2000` only for a short test.

Main evidence directories: `frame4_fused_full14201`, `control_full14201`,
`opencv_fused_full14201`, `startup_optimized`, `startup_opencv_fused` and
`startup_control`. These contain actual summaries,
thermal samples and per-frame observations/trial logs, not estimated timings.
