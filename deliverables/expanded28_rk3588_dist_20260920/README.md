# Expanded-28 Detector + Dist Public Code + GMC / RK3588S

## Model and Deployment

This deployment uses the **28-video Frozen-P3 + Add-on P2** checkpoint, not the
older September 4 checkpoint. No detector retraining was performed for deployment.

Original checkpoint on server 47:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_addon/p2/weights/best.pt
```

Exported model on server 47:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/expanded28_rk3588_dist_20260920/detector_960x544_int8.rknn
```

Board code: `/home/orangepi/ultralytics_yolov8_ziye`.
Board deployment: `/home/orangepi/deployments/expanded28_dist_20260920`.
The RKNN file is in that deployment directory. Local copy is beside this README.
`model_manifest.json` records source checkpoint and RKNN SHA256, calibration
sources, compiler/runtime versions and upstream tracker commit.

The board was identified from its live device tree as **RK3588S OPi CM5**,
Ubuntu 22.04, kernel `6.1.99-rockchip-rk3588`, runtime **2.3.2**.
Its Ethernet IPv4 is `192.168.123.10`. During deployment the Mac was on a different
IPv4 subnet; SSH worked through IPv6 link-local without changing either network:

```bash
ssh -6 'orangepi@fe80::d536:9cee:9054:6299%en7'
```

The `%en7` interface suffix is specific to this Mac's Ethernet interface.
No boot service, kernel, network setting or thermal protection was changed.

## Measured Results

These are the original unoptimized baseline measurements, not the current speed
limit. The optimized deployment and new measurements are documented in
`../expanded28_dist_opt_20260920/README.md`. The identical detector subsequently
reached 86.82 FPS over 2,000 frames with big-core scheduling and one OpenCV thread.

All rows include CPU video decoding, letterbox/color conversion, native RKNN
inference and DFL/NMS. Tracking rows additionally include real, per-frame GMC and
ordered association. Drawing, display, camera transport and encoding are excluded.

| Configuration | Frames / excluded warmup | Measured FPS |
| --- | --- | ---: |
| Detector, NPU core 0 | 2,000 / 100 | 24.66 |
| Detector, three independent NPU cores | 2,000 / 100 | 49.29 |
| Detector + public Dist + GMC, three NPU cores | 14,201 / 100 | 18.36 |

The last 2,201 frames of the full run averaged **16.28 FPS** after sustained heat.
These are measurements on the board as connected, not its theoretical maximum.
The detector-only runs use the first 2,000 frames; the full-run average covers
different scene complexity and a much longer thermal history.

Three fresh-process startup measurements (including interpreter/imports, loading
all three NPU contexts, decoding and processing frame 0) were **716.6, 753.7 and
734.5 ms**; median **734.5 ms**. After model initialization, first-frame processing
was **122.9, 155.3 and 126.1 ms**; median **126.1 ms**. These are not power-on boot
times, and first-frame completion does not require a visible target.

During the full run, average decode was **7.54 ms/frame**, native preprocessing
**12.81 ms**, per-worker NPU run **41.14 ms**, postprocessing **1.14 ms**, and
tracker + GMC **46.73 ms**. Stages overlap; their sum is not the pipeline period.
Average read-to-ordered-result latency was **326.7 ms**, including the bounded
six-frame queue. This is distinct from first-frame latency and reciprocal FPS.

The SoC reached approximately **85 C** (some CPU sensors approximately 87 C).
NPU frequency was **800 MHz in 76 of 79 samples**, versus the configured maximum
1 GHz. Big-core policy4 was mostly 1.608 GHz and sometimes fell to 408 MHz.
The kernel reported fan PWM 255; the user subsequently confirmed a heatsink is
fitted but there is **no physical fan**. CPU/GMC cost and thermal throttling both limit this
result; it is not comparable to the older detector/RK-BoT-SORT-only 85 FPS figure.

The full INT8 run completed without dropped frames: 8,857 detections and 8,161
displayed current-frame track observations. Detector outputs for the first 2,000
frames were **bit-identical** across single-core, three-core and tracked runs.
This validates execution consistency, not detector precision/recall against GT.
See `benchmark_comparison.json` and the per-run `summary.json` files in
`final_detector_1/`, `final_detector_3/`, `final_pipeline_three_full/`, and
`startup_process_launch/`.

## Runtime and Reproduction

- Detector: native C++ RKNN INT8, fixed **960 x 544 (W x H)**, four output scales.
- Input: BGR video decoded with OpenCV, resized/letterboxed to RGB, padding 114.
- Thresholds: `conf=0.03`, NMS IoU `0.45`, maximum 100 detections.
- Three independent RKNN contexts use NPU cores 0, 1 and 2, one worker per core.
- Results are consumed in original frame order; the queue is bounded to six frames.
- Tracker: unchanged public Dist repository BOTSORT implementation, no ReID.
- Tracker thresholds: high 0.03, low 0.01, new-track 0.10, match 0.8, score fusion
  disabled; buffer 30 scales to 100 frames at the source's 100 FPS.
- GMC: actual causal `sparseOptFlow`, downscale 2, every original video frame.
- Only confirmed, current-frame associated detections are displayed/output.
  No GT, extrapolated boxes, area filter or cached detection/GMC is used in timing.

This is the **Dist public-code adaptation + GMC**, not a claimed implementation
of the paper's FLIT/L2-IoU features absent from the executed repository entry point.
The NumPy loader avoids the upstream package's unrelated detector imports. It
loads tracker modules unchanged and extracts two upstream utility functions.
The board does **not** need PyTorch. PyTorch was used only on server 47 to export
the `.pt` checkpoint. Association uses NumPy/SciPy/lap; GMC uses CPU OpenCV.

The loader was checked against the previous full public-code replay on all
**14,201 frames**: **8,304 displayed observations**, identical IDs and detector
associations. See `dist_adapter_equivalence.json`. That cached replay is a
correctness check only, not the board FPS benchmark.

Run the installed deployment on the board:

```bash
cd /home/orangepi/deployments/expanded28_dist_20260920
bash run_on_board.sh --workers 3 --warmup 100 \
  --output "$PWD/results/my_full_run" --save-observations
```

Use a new output directory for each run. Omit `--save-observations` when frame
JSON output is unnecessary. To benchmark detection alone:

```bash
bash run_on_board.sh --workers 3 --detector-only --frames 2000 --warmup 100 \
  --output "$PWD/results/my_detector_run"
```

For one NPU core, use `--workers 1 --core 0`. Do not run benchmark processes
concurrently. The wrapper sets `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

Rebuild the native shared library:

```bash
bash /home/orangepi/ultralytics_yolov8_ziye/scripts/anti_uav/build_dist_detector_on_board.sh \
  /home/orangepi/deployments/expanded28_dist_20260920/libanti_uav_detector.so
```

The deployment has an isolated `venv` using system OpenCV 4.5.4 plus NumPy 1.24.4,
SciPy 1.10.1 and lap 0.5.12. Offline ARM64 wheels are in `wheels/` on the board.
Upstream sources and their license are in `Dist-Tracker/` on the board.
Application changes were pushed from the Mac and pulled on server 47 and the board;
Git bundles were used for the offline board. Existing local native-tracker edits
were not included in this deployment.

## Measurement Definitions

The input is the complete original **Video00009**, 1920 x 1080 HEVC,
14,201 frames at 100 FPS. Source playback FPS is not inference throughput.
The pipeline benchmark processes every frame, not a sampled validation subset.

`summary.json` in each result directory records:

- `steady_fps`: completed outputs / wall time after the first 100 warmup outputs.
- `first_result_from_python_entry_ms`: fresh Python entry to first completed frame,
  including imports, opening the video, loading all contexts and first processing.
- `first_frame_read_to_result_ms`: first frame read to completed result, after
  model initialization. The first video frame contains no detected target; an empty
  result is still a completed frame, not a promise of a visible box.
- `stages_ms`: decode, preprocessing, NPU run, postprocessing, tracker + GMC and
  ordered frame latency. Overlapping stage times must not be added to infer FPS.

Startup trials are fresh processes with an already-running OS and potentially
warm filesystem caches, not power-on boot-to-result measurements. Timing excludes
camera exposure/transport, display, drawing and video encoding. No preloaded frames
are used in the final video benchmarks. Per-frame JSON writing is included where
`save_observations=true`.

`startup_process_launch/summary.json` uses an external parent process to time
launch-to-first-result, including shell and interpreter startup; this is the
source of the 734.5 ms median above. The lower-level per-run Python-entry timing
is retained separately rather than conflated with process-launch timing.

Temperature and clocks were recorded without disabling throttling. The board
reached thermal limits under sustained work; do not interpret a short NPU-only
peak or an earlier RK-BoT-SORT number as this pipeline's sustained throughput.
`final_full_thermal_clock_samples.txt` contains repeated four-line samples:
board Unix time, zone-0 temperature (milli-C), NPU Hz, big-core policy4 kHz.

Successful INT8 export and functional tracking do not establish identical PT/INT8
accuracy. The compiler reported a weight-outlier warning, preserved in
`build_rknn.log`. The held-out Video00004/Video00009 were not used for calibration;
384 images from six older training videos were used. No new mAP claim is made here.
