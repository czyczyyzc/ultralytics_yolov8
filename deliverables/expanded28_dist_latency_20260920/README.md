# Read-frame and Preprocessing Latency Work

## Implemented, not yet board-validated

The stable deployment is unchanged. Two new opt-in native C++ paths are ready
for isolated testing once board SSH access is restored:

1. `--decoder ffmpeg`: direct software H264/HEVC decoding with one decoder
   thread. It does not use MPP, skip frames or force a different display order.
   The intent is to test decoder frame-thread startup buffering separately from
   inference. Both launch-to-first-result and read-to-first-result must be
   measured; moving work into initialization is not an end-to-end improvement.
2. `--preprocess fused`: combine exact half-size downsampling and BGR-to-RGB
   conversion into one ARM NEON pass. For the current 1920x1080 source this
   produces the same 960x540 content in the 960x544 letterboxed input. Padding is
   reused when the layout is unchanged; the final contiguous copy into RKNN
   memory is retained. Other resize ratios use the existing OpenCV fallback.

This is not a zero-copy camera pipeline. No resolution, thresholds, tracking
rules or inference model are changed. Defaults remain OpenCV.

## Evidence and limitations

- ARM64/NEON kernel: 132 randomized dimensions/strides plus five invalid-input
  cases passed on the local Mac with AddressSanitizer and UndefinedBehaviorSanitizer.
- Scalar kernel on server 47: 264 random color/grayscale layouts exactly matched
  OpenCV 4.11.0 resize(INTER_LINEAR) followed by BGR-to-RGB, including untouched
  destination guard regions. See `opencv_equivalence.json`.
- These are correctness tests, not RK3588 timing benchmarks. Board OpenCV 4.5.4
  tensor equivalence and the complete rebuilt detector library still need testing.
- The direct software FFmpeg path has not yet been compiled/tested on the board.
- SSH still stalled at handshake in this turn. USB ADB listed no connected
  devices. No new board benchmark was started and no new board FPS/latency is
  claimed. The last stable first-frame read/decode measurement remains 56.25 ms;
  input preprocessing remains 4.20 ms for that same first-frame trial.

The OpenCV 4.5.4 upstream capture implementation sets the codec thread count
independently from OpenCV's image-operation threads. Thus `cv::setNumThreads(1)`
does not establish single-thread FFmpeg decoding. Whether this explains the
board's first-read overhead is a hypothesis, not a completed diagnosis:
https://github.com/opencv/opencv/blob/4.5.4/modules/videoio/src/cap_ffmpeg_impl.hpp

## Resume on the board

After restoring access, pull the committed code, inspect the previous MPP crash
without rerunning MPP, and create a new deployment directory. Compile with
`scripts/anti_uav/dist_native/build.sh` and
`scripts/anti_uav/build_dist_detector_on_board.sh`. These require the board's
existing development headers and runtime libraries. Disable core dumps in the
diagnostic process (`ulimit -c 0`), not globally.

Start with the independent `decode_probe VIDEO ffmpeg 100`, then compare:

- Baseline OpenCV decode + OpenCV preprocessing.
- OpenCV decode + fused preprocessing.
- Single-thread software FFmpeg decode + OpenCV preprocessing.
- Single-thread software FFmpeg decode + fused preprocessing.

Measure three fresh-process first-frame trials and full 14,201-frame throughput,
record temperatures, and compare every output against the stable reference.
Use the unchanged stable deployment until these checks pass. RGA/MPP remain
separate experiments, not prerequisites for these two software optimizations.
