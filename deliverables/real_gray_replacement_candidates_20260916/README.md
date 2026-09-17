# Isolated Grayscale Replacement Batch

## Server Location

Server: `root@47.107.185.207`.

Output directory:

```text
/mnt/andrew/anti_uav_model_refinement/data/real_gray_replacement_candidates_20260916
```

Source training pool:

```text
/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_28videos_20260916
```

Script:

```text
/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/build_gray_replacement_batch.py
```

The batch was launched at 2026-09-16 17:48:32 +0800, PID 2363872, using an
independent `.venv_gray_synthesis` environment, CPU affinity 126,127 and nice 15.
It used no GPU and completed at 2026-09-16 18:08:09 +0800 after about 19.6 minutes.
The Frozen-P3 + Add-on P2 training run was not restarted or modified. These
synthetic candidates were not used in that run.

Final output: 856 accepted grayscale images, 856 YOLO label files, 856 edit masks
and 16 comparison previews. There are 219 distinct replaced source frames from
17 training videos, using all 53 eligible catalog IDs. The generator selected 360
source frames from 18 videos; unsafe repairs and duplicate outputs were not saved
as new training images. This is not 856 independently captured frames.

`summary.json`, `manifest.json`, `train_synthetic.txt` and `status.json` now contain
the final result. `verification_snapshot.json` is the older partial 211-image check,
not the final count. Local previews and the contact sheet have been synchronized.

Source-size strata use the original bbox short edge after hypothetical 960x544
letterboxing, not the new rendered bbox or the detector report's long-edge bins:

| Source short edge | Accepted variants |
| --- | ---: |
| <=8 px | 381 |
| >8 to 16 px | 240 |
| >16 to 32 px | 145 |
| >32 px | 90 |

## Selection And Output

- Only current training-list images with matching approved video identities and
  annotations are eligible. Entire validation and test video hashes are excluded.
- Select up to 20 source frames per usable video, balanced across target-size
  strata. Create up to four asset variants per selected frame.
- Use 53 native-alpha cutout candidates from the supplied catalog. IDs 24 and 25
  contain remote controls and are excluded. Opaque photographs are not used.
- Reconstruct local background from an approved neighbor in the same video, with
  registration, brightness matching and a feathered foreground contour. There is
  no rectangular independent-noise fallback.
- Preserve target center and geometric long edge; retain each new asset's aspect
  ratio. Recompute its normalized YOLO bbox from the rendered alpha mask.
- Save only successful replacements in `images/` and `labels/`. Failed candidates
  and their reasons are recorded in `manifest.json`, not duplicated into training.
- All PNGs are grayscale, lossless and unannotated. Edited-region masks are in
  `masks/`; the renderer adds 1px corner boxes to separate `previews/` only.
- `train_synthetic.txt` lists accepted positives only. It is not a balanced
  standalone dataset, and is not appended to any active training list.

The synthesis method supports original-frame target short edges >=3 pixels and
long edges <=160 pixels. This is a synthesis eligibility restriction, not a new
detection filter or a deletion of large original training targets. Older image
paths without a matching approved registry are left untouched.

This is an offline detector-augmentation candidate batch, not a temporally
consistent tracking sequence. Pixel checks do not establish perfect visual
blending, camera realism, valid asset licensing or improved detector accuracy.
Review before a separate controlled training experiment with fixed sampling.

## Verification

20 CPU tests passed both locally and on server 47. They cover held-out exclusion,
uncertain/unreviewed-frame rejection, current-list membership, deterministic size
sampling, updated boxes, success-only export, no overwrite and background repair.
These are isolated synthesis tests, not the complete YOLO test suite.

The generator checks source and catalog hashes, saved PNG equality, bbox
normalization, pixel identity outside edit masks and training-configuration hashes.
Local previews are copies of the server files, not new synthesis runs.

On 2026-09-17 a separate full audit reread all 856 output/source images, masks
and labels. All hashes and updated boxes matched; decoded pixels outside every
mask were unchanged; no held-out video hash was present; protected training
configuration hashes were unchanged. See `verification_final.json`. This audit
does not replace manual visual review or evaluation of training benefit.
