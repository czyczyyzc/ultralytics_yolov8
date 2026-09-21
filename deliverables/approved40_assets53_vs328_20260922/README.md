# Controlled Cutout-Catalog Experiment: 53 vs 328

## Experiment

Both arms use the same 40 training videos plus the preserved Anti-UAV300 RGB
schedule, the same initial P3 checkpoint, seed, batch size, optimizer settings,
YOLO transforms and gray validation/checkpoint-selection rule. Neither arm is
resumed from the other's partially trained model.

| Setting | Control | Expanded candidates |
| --- | --- | --- |
| Enabled cutout entries | 53 | 328 = 53 original + 275 screened candidates |
| GPU | 0 | 1 |
| P3 stage | 15 epochs | 15 epochs |
| Frozen-P3 add-on P2 stage | 15 epochs | 15 epochs |
| Batch / input | 64 / 960x544 | 64 / 960x544 |
| Positive / negative epoch slots | 143771 / 25371 | 143771 / 25371 |
| Online replacement attempt probability | 0.5 | 0.5 |
| Eligible cached positive frames | 54926 | 54926 |

Asset choice changes; original-versus-replacement scheduling does not. Existing
background packs, original images, labels, validation list, repetitions and
negative pool are reused without modification. Rejected composites retain the
original image/box in both arms, so the realized replacement success rate can
vary with catalog contents and is part of the treatment.

This experiment includes every screened candidate, not all raw workbook images.
Rejected candidate IDs 252/253 (manual cover/line drawing), 258 (truncated
aircraft), and 295 (charger contamination) remain excluded, as do original
disabled IDs 24/25. Candidate masks are experimental, not pixel-perfect or
production-qualified. Duplicate catalog entries are retained, not deduplicated
into distinct aircraft models.

## Server Locations

Server: `root@47.107.185.207`.
Repository: `/mnt/chenziye/codes/ultralytics_yolov8`.

Control run:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online53_20260921`

Expanded run:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922`

Expanded read-only background view and enabled catalog:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_online_replacement_40videos_assets328_20260922`

The enabled catalog is `assets/catalog.json`; `asset_ablation.json` records
provenance. Do not remove the original 37/40-video cache directories: the new
view links their immutable packs and original frame paths.

## Validation and Results

Before training, all cutouts pass checksum/alpha checks. A six-frame, all-asset
smoke test exercises compositing, updated bboxes and the real multiworker YOLO
loader. A further 328-frame seeded preview must contain at least one successful
composite for every asset; failures fall back to another audited source frame.
This automated QA does not guarantee realism or improved accuracy.

The original 53-asset run continues unchanged. The expanded run waits for the
control to finish before comparing finalized checkpoints. `Video00009`'s same
1421-frame validation subset selects weights. `Video00004` is test-only and is
never used for augmentation, mining or weight selection.

Final comparison: both P3 models and both frozen-P3/add-on models, fixed 960x544,
FP32 detector evaluation, NMS IoU .45, match IoU .5, conf .01/.03/.05. Report
precision, recall, FP, mAP50 and 4-8px recall. These are not RKNN INT8 or tracking
metrics. The matched-epoch experiment isolates changing the asset catalog, but
a single training seed is not a significance test.

Under the expanded run:

- `experiment_status.json`: QA, training, reference waiting, evaluation or failure.
- `status.json`: current training stage/epoch.
- `augmentation_smoke/summary.json`: all-asset compositing smoke results.
- `augmentation_preview/index.html`: 328 original/replacement/bbox examples.
- `training_p3/p3/results.csv`, `training_addon/p2/results.csv`: losses/metrics.
- `training_p3/p3/weights/best.pt`, `training_addon/p2/weights/best.pt`: created by training.
- `COMPARISON.md`, `comparison/results.json`: created only after final evaluation.

No result is implied by creating these experiment directories. Runtime status
and completed evaluation files are authoritative.
