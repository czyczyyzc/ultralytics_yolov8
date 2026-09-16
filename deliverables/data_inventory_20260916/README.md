# Training Data Inventory

Snapshot: 2026-09-16 16:48:57 +0800, server 47.107.185.207.
The approval directory was actively growing: an earlier scan found 22 tasks,
and this frozen snapshot contains 23. Figures below refer to this snapshot.

## Used Training Data

| Measure | Old manual-clips0123 model | Current native-only schedule |
| --- | ---: | ---: |
| Gray training videos | 10 | 14 |
| Distinct gray image paths | 23,434 | 26,253 |
| Distinct Anti-UAV300 RGB image paths | 35,773 | 35,736 |
| Distinct total image paths | 59,207 | 61,989 |
| Training entries per epoch, including repetitions | 86,505 | 89,748 |

Current entries: 53,975 gray and 35,773 RGB, including sampling repetitions.
Across both sources: 76,436 positive entries and 13,312 negative entries (14.83%).
Hard-negative substitutions can lower the number of distinct negative paths while
preserving the number of negative training entries and all original positive
repetitions. Distinct image paths are not a pixel-hash or near-duplicate audit.

Native video00009 validation remains 1,421 sampled frames, plus 64 synthetic
diagnostic views. Video00004 remains test-only. Restoring the original augmentation
does not remove new native videos or labels. The old 480 and 1,592 extra views
were generated samples, not newly captured images.

## Newly Available, Not Yet Used

`/mnt/andrew/anti_uav_model_refinement/data/approved_tasks/` now contains
23 distinct approved video tasks, compared with 9 used in the prior dataset build.
These approved videos contain 105,783 raw frames in total.

The 14 additional videos contain:

- 57,246 raw frames.
- 57,145 included/reviewed frames.
- 31,605 positive frames and 31,605 COCO boxes.
- 25,540 reviewed negative frames.
- 101 excluded frames, not training candidates.

All 14 corresponding source video files were found under
`/mnt/andrew/video-labeler/videos/`. They have not been extracted into or added to
the current training schedule. Full names and task paths are in
`new_approved_audit.json`. The additional left-translate video00009 is a different
video from the held-out sunny stationary video00009; identify videos by full name
and hash rather than the trailing video number.

The 23 approved videos and seven archived old videos have no overlap by manifest
SHA256, giving 30 distinct video files across the two annotation collections.
Keeping one validation video and Video00004 as test would leave 28 training
candidates, subject to source/annotation/leakage checks and explicit inclusion.
Different hashes do not rule out temporally overlapping or re-encoded clips.

Verification: manifest frame inclusion consistency, COCO image counts and source
video existence. Video SHA256 values were compared from manifests; raw videos
were not rehashed or fully decoded in this inventory. Adjacent frames are not
independent observations, and not all reviewed negative frames should be added
indiscriminately to a training schedule.

## Restoration Status

The online experiment was stopped at the user's request. Native-only configuration:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_augmentation_restored_20260916/train_hardneg_gray_monitor.yaml`.

Train and validation lists were verified byte-identical to the native lists of the
online experiment. There is no `online_scale` key and no `zoom_train` entry.
No new long training job was launched and no deployment weights were replaced.
