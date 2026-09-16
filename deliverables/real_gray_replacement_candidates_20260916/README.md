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
It uses no GPU. Check the server `status.json` for completion. The current
Frozen-P3 + Add-on P2 training run is not restarted or modified.

The first independently checked snapshot contains 211 accepted images from
three training videos, using 52 distinct catalog IDs. All 211 image and label
hashes passed; a spread of saved images was also decoded again to check the
unchanged background. This is a partial snapshot while generation continues,
not the final batch count. `verification_snapshot.json` records this check.
On completion, the server writes `summary.json`, `manifest.json`,
`train_synthetic.txt` and a complete `status.json`.

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
