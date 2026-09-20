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

## Runtime and Reproduction

- Detector: native C++ RKNN INT8, fixed **960 x 544 (W x H)**, four output scales.
- Input: BGR video decoded with OpenCV, resized/letterboxed to RGB, padding 114.
- Thresholds: `conf=0.03`, NMS IoU `0.45`, maximum 100 detections.
- Three independent RKNN contexts use NPU cores 0, 1 and 2, one worker per core.
- Results are consumed in original frame order; the queue is bounded to six frames.
- Tracker: unchanged public Dist repository BOTSORT implementation, no ReID.
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

Temperature and clocks were recorded without disabling throttling. The board
reached thermal limits under sustained work; do not interpret a short NPU-only
peak or an earlier RK-BoT-SORT number as this pipeline's sustained throughput.
`final_full_thermal_clock_samples.txt` contains repeated four-line samples:
board Unix time, zone-0 temperature (milli-C), NPU Hz, big-core policy4 kHz.

Successful INT8 export and functional tracking do not establish identical PT/INT8
accuracy. The compiler reported a weight-outlier warning, preserved in
`build_rknn.log`. The held-out Video00004/Video00009 were not used for calibration;
384 images from six older training videos were used. No new mAP claim is made here.
