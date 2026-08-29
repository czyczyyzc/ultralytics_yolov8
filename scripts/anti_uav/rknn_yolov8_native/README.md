# Native RKNN YOLOv8 + RK-BoT-SORT

The current RK3588S deployment baseline is:

- YOLOv8n, fixed `960x544` (`WxH`) input
- RKNN INT8 model built with RKNN-Toolkit2 2.3.2
- native RKNN C API with zero-copy input/output buffers
- three ordered detector workers, one per RK3588S NPU core
- detector-based RK-BoT-SORT without ReID, IMU, or camera pose input

See the complete Chinese handoff guide at
[`docs/anti_uav_rk3588s_deployment_cn.md`](../../../docs/anti_uav_rk3588s_deployment_cn.md).

## Build on the board

Install a matching `librknnrt.so` and make `rknn_api.h` available, then run:

```bash
sudo apt-get install -y build-essential pkg-config libopencv-dev

RKNN_INCLUDE=/path/to/rknn/include \
RKNN_LIB_DIR=/path/to/rknn/lib \
./build_on_board.sh
```

`RKNN_INCLUDE` can be omitted when `rknn_api.h` is installed under a standard
system include directory. `RKNN_LIB_DIR` can be omitted when `librknnrt.so` is
already installed in the system library path.

## Three-core real-time pipeline

```bash
./build/native_yolov8_video MODEL.rknn VIDEO.mp4 \
  --workers 3 \
  --core-mask 0_1_2 \
  --worker-cpu-base 4 \
  --queue-size 3 \
  --tracker rk_botsort \
  --source-fps 107 \
  --conf 0.25 \
  --track-high-thresh 0.25 \
  --track-low-thresh 0.10 \
  --new-track-thresh 0.30 \
  --track-buffer-sec 1.0 \
  --track-prediction-sec 0.0 \
  --output-json output/benchmark.json \
  --predictions-csv output/detections.csv \
  --tracks-csv output/tracks.csv
```

The three workers own separate duplicated RKNN contexts. The implementation
binds them to NPU cores 0, 1, and 2 and pins their CPU-side work to cores 4, 5,
and 6. Detection results are reordered by frame index before tracking, so
RK-BoT-SORT always consumes chronological input.

Use `--queue-size 3` for bounded live-stream latency. `--queue-size 12` is only
for offline maximum-throughput benchmarks and consumes substantially more frame
memory.

The current preprocessing implementation uses OpenCV resize and BGR-to-RGB
conversion while writing directly into RKNN input memory. RKNN input and native
INT8 output buffers are zero-copy; RGA preprocessing is not enabled in this
source tree.

## Measured reference performance

The previous same-shape `960x544` release model measured on Orange Pi CM5
RK3588S with performance governors enabled:

- one NPU context: 47.62 FPS
- three contexts, detector only, queue 3: 129.41 FPS
- three contexts plus RK-BoT-SORT, queue 3: 129.57 FPS

These numbers include video read for the three-worker test. Re-run the benchmark
for every final model artifact, camera backend, kernel, and runtime combination.
