# Native Pipeline Queue and Hardware I/O Experiments

## Verified queue sweep

Same RK3588S, 960x544 INT8 model, conf 0.03, three split-core RKNN workers,
native Dist and per-frame compact GMC. Existing stable executable was used.
Each run processes the first 2,000 Video00009 frames, excludes 100 warmup frames,
and records all detector/tracker outputs. Run order: 9,3,4,6,6,4,3,9.

| Maximum in-flight frames | Mean FPS, two runs | Mean read-to-result ms | Mean of per-run P95 ms |
| --- | ---: | ---: | ---: |
| 3 | 73.87 | 40.15 | 46.43 |
| 4 | 84.29 | 46.05 | 56.29 |
| 6 | 84.92 | 46.19 | 56.43 |
| 9 | 85.04 | 46.15 | 56.61 |

All eight runs exactly match the reference on detections, displayed boxes/IDs
and GMC matrices. No dropped frames or changed thresholds. Results are short-run
video-file measurements, not sustained camera-to-display latency. Temperature
was recorded but not controlled. P95 values above are means of two run-level
P95 values, not a pooled percentile.

Conclusion: reducing the limit from nine to four does not materially reduce
latency in the current ready-worker pipeline. A limit of three saves about 6 ms
of mean latency but loses about 13% throughput. Keep the stable default at nine
for maximum throughput; select three only when that measured tradeoff is wanted.
No configuration is advertised as improving both speed and latency from this
sweep alone. Detailed numbers: `queue_comparison.json`, `q*/summary.json`.

## Experimental hardware path: not approved for deployment

The board has MPP/RGA libraries, headers, `/dev/mpp_service`, `/dev/rga`, DMA heap
nodes and FFmpeg's `hevc_rkmpp` decoder. An opt-in native FFmpeg/RKMPP source and
opt-in RGA resize mode were added and compiled successfully on the board.
Default decoding/preprocessing remain OpenCV. No silent fallback to CPU is used
when the hardware path is explicitly selected.

The first combined RKMPP/RKNN trial (`mpp_q4_short`) exited with segmentation
fault (139), without a valid benchmark summary. Subsequently SSH handshakes
stalled while ICMP still responded. The crash log/backtrace could not yet be
retrieved. The cause, and whether it explains the SSH failure, are not established.
There is **no valid MPP FPS or quality result**. RGA has been compiled but **not
yet run**. Neither hardware option is selected in the stable deployment.

Experimental binaries/logs are under
`/home/orangepi/deployments/expanded28_dist_io_20260920/`.
The validated native deployment remains untouched under
`/home/orangepi/deployments/expanded28_dist_native_20260920/`.

The RKMPP implementation currently maps decoded NV12 DMA buffers and uses CPU
color conversion into BGR. The RGA experiment uses virtual-address image buffers
for resize, then the existing RGB copy into RKNN input. This is **not** an
end-to-end zero-copy pipeline and must not be described as one.

## Recovery and next validation

No further hardware experiments are running or scheduled automatically. Restore
board access first and inspect the crash log/core and kernel/storage status.
`scripts/anti_uav/dist_native/decode_probe.cpp` isolates video decoding without
loading RKNN or allocating NPU buffers; it is added for the next diagnostic step
and has not yet been compiled/tested on the disconnected board. Disable core
dumps for bounded diagnostic processes to avoid large device-buffer dumps.

Validate CPU decoding first, then isolated RKMPP, then combined operation. Test
RGA separately. Check complete-video detection and tracking quality before
selecting any hardware path; different conversion/interpolation rounding may
change small-object predictions even at the same resolution and threshold.
