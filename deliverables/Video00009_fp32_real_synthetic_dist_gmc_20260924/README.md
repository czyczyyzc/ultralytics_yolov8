# Video00009 real/synthetic FP32 Dist+GMC visualization

Date: 2026-09-24. This is a server-side offline visualization, not an RK3588
runtime or tracker-accuracy benchmark. "GMT" in the request was interpreted
as the previously used Dist tracker with GMC camera-motion compensation.

## Videos

- `Video00009_real_FP32_960x544_Dist_GMC_conf003.mp4`: all 14,201 original
  consecutive Video00009 frames.
- `Video00009_synthetic_FP32_960x544_Dist_GMC_conf003.mp4`: the same 14,201
  frame timeline, with only the 565 successfully replaced frames from the
  evaluation-only paired synthetic set substituted. All other frames use
  the original video; the synthetic set itself contains only 1,421 selected
  frames and is **not** a continuous synthetic clip.
- `Video00009_synthetic_565_selected_FP32_Dist_GMC_10fps.mp4`: only the 565
  successfully replaced frames, selected from the already-rendered synthetic
  video in original frame-index order. This is a 56.5-second, 10 FPS viewing
  montage, not a continuous 10 FPS capture. Original frame indices remain
  visible in the video overlays. Detector, GMC, and tracker were **not** rerun
  for this cut; its boxes and IDs are inherited from the complete 14,201-frame
  causal run. SHA256:
  `17ab2a4e528da96ba5f36ccdd6f1ce83042f27720cf2997cac59a9252537035d`.

The two full-length videos are H.264, 1600x784, 100 FPS playback, 142.01 seconds. The 100 FPS
is source playback rate, **not** detector or tracking throughput. Each stream
was tracked independently, without carrying the original stream's IDs into
the synthetic stream. Only confirmed, currently observed detection boxes are
shown: 1-pixel corner boxes on clean-source crops, no GT, no crosshair,
no predicted/interpolated boxes or ID remapping. Replaced synthetic frames
are labeled in the video.

## Inference protocol

- Detector: Frozen-P3 + Add-on P2, 40 approved videos + 328 transparent
  target assets, 50% training replacement; checkpoint on 47 server:
  `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922/training_addon/p2/weights/best.pt`.
  Checkpoint SHA256: `6aec79be8e1562d5cbab4f7b1427d156edbd4fff8dba3095302b17b85aac6472`.
- PyTorch FP32, detector input 960x544 (W x H), conf=0.03, NMS IoU=0.45,
  max_det=100. Original frames were inferred once; synthetic replacements
  were inferred separately. Unmodified frames reuse exactly the same
  detection observations in both streams.
- Tracker: pinned Dist-Tracker public-code BoT-SORT adaptation with the
  established low-score/no-score-fusion settings; causal compact
  `EfficientGMC(width=320, corners=128, refresh=5, resize_first=True)`.
  This is not a claimed implementation of the full Dist paper's L2-IoU term.
- Original video SHA256:
  `2629efe7da36dfd079985344653a7a4ecaa25f651954e5258a53c50e960d6b86`.
  Synthetic dataset: `/mnt/andrew/anti_uav_model_refinement/external_eval/synthetic_gray_pair_20260922/`.

The FP32 pass covered 14,201 frames in each stream, with 9,243 original
and 9,239 synthetic detections at conf=0.03. Of the 565 substituted frames,
562 changed the detector output. Tracking covered all frames, with 8,323
original and 8,315 synthetic displayed observations. These are raw counts,
not TP/FP or identity-quality measurements.

## Limitations and evidence

The 328 source cutouts were used in training, and Video00009 participated in
checkpoint selection; this is not an unseen-type or independent test.
Randomly selected synthetic aircraft change between sampled frames, so this
video cannot establish temporal appearance consistency or ID-switch quality.
The full synthetic video is intentionally mostly original footage: only
565 / 14,201 frames carry a replacement. Individual replacement frames are
marked for inspection.

Server result directory:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/Video00009_real_synthetic_fp32_dist_gmc_20260924/`.
It contains paired detector JSONL, independent tracker JSONL, stage summaries,
video summaries, the two MP4s and their previews. Script:
`scripts/anti_uav/render_fp32_dist_synthetic_video.py`.
The local videos have the same SHA256 as their server versions:

- Real: `51041491bd568af292ea537bd8b938a92de4240090e580a153338e88f3cea933`
- Synthetic: `85594d233bfaf158f067f9a16afff70032590db0f5d3a2e781eade13b4702bd4`

Three related unit tests passed locally and on 47 server. Detector, track,
and encoded-video frame counts were checked; both encoded videos were probed
at 14,201 frames / 142.01 seconds. The first successful synthetic frame
(frame 1060) was visually inspected against its real counterpart.
