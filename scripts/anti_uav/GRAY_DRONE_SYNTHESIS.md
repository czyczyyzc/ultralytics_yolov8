# Grayscale Drone Replacement

CPU-only, offline frame augmentation prototype. It never modifies an input image,
the original workbook, a training list, or a running training process. Outputs are
visual-review candidates, not automatically approved training data.

## Run

From the repository root, use a Python environment with the dependencies in
`scripts/anti_uav/requirements-gray-synthesis.txt`. No GPU or model download is
needed. Python 3.10 or newer is required.

```bash
python scripts/anti_uav/synthesize_gray_drone_replacements.py catalog \
  --xlsx '/Users/czyczyyzc/Downloads/无人机机型汇总(1).xlsx' \
  --output deliverables/my_drone_synthesis/catalog

python scripts/anti_uav/synthesize_gray_drone_replacements.py synthesize \
  --catalog deliverables/my_drone_synthesis/catalog/catalog.json \
  --source-manifest deliverables/self_collected_gray_training_samples_20260914/manifest.json \
  --asset-ids 1,330,332 \
  --temporal-registry /path/to/training_video_registry.json \
  --variants 3 --preview-count 10 --seed 20260914 \
  --output deliverables/my_drone_synthesis/samples
```

Both output directories must be new. Existing outputs are never overwritten.
For the local verified run, the Python executable is
`.codex_work/gray-synthesis-venv/bin/python`. That local environment is not part of Git.
On the server use its own Python environment and server-local input paths.

The two commands can be run consecutively in a job. Catalog extraction is needed
only once per workbook revision. Synthesis uses the resulting catalog entirely
offline. `--asset-ids` uses the actual Excel sequence IDs, not ZIP media numbers.
Within each video and variant the asset is fixed; subsequent variants cycle
through the selected asset list. Omitting `--asset-ids` uses all eligible native
alpha candidates, including ones that may still need semantic review.

## Inputs

The source manifest is the existing training-sample export format:

```json
{
  "samples": [
    {
      "video": "Video00001",
      "image": "images/Video00001/001194.jpg",
      "label": "labels/Video00001/001194.txt",
      "sha256": "SHA256 of the original image file"
    }
  ]
}
```

Paths are relative to the manifest directory, or `--source-root`. Each label is
normalized YOLO `0 cx cy width height`; an empty label is a genuine negative frame.
Source SHA256 is mandatory; file existence, bounds, unique images and path
traversal are checked. Label hashes are recorded in the output manifest.

`Video00004` identities are always rejected. Additional held-out names can be
supplied with repeated `--exclude-video`. This is an identity/name guard, not
proof against renamed or overlapping clips. Only supply an already audited
training-only manifest; never use held-out frames as synthesis backgrounds.

The default background mode is now `temporal`. Supply exact original training
videos and their approved manifest/COCO annotation files in a registry:

```json
{
  "videos": [
    {
      "video": "original_training_clip.mp4",
      "approved_manifest": "approved_manifest.json",
      "coco": "coco/annotations.json"
    }
  ]
}
```

Registry paths are relative to the registry file, or absolute. This uses the
video-labeler approved export schema: `video.sha256`, `frames.indexBase=0`, included
and excluded frame indices, hashed `files`, and COCO images with `frame_index`.
COCO `image_id` is NOT assumed to equal the zero-based video frame index. Both
video and COCO hashes are verified against the approved manifest. Donors must be
from the exact same original video and reviewed frames; uncertain/unreviewed and
held-out frames are excluded. The source image must match the decoded frame and
its box must agree with that frame's approved annotation.

If a registry entry or a safe donor is unavailable, the original sample is kept
with a skip reason. There is no automatic plane/noise fallback. The old method
can only be requested explicitly with `--background-mode legacy-plane` for
ablation. It can create visible rectangular texture patches and should not be
used as training-quality synthesis.

## Processing

1. Match embedded Excel pictures to metadata by drawing anchors/relationships.
2. Save all original pictures and extract conservative native-alpha cutouts.
   Opaque pictures are marked `opaque_requires_segmentation` rather than guessing
   which item in a product bundle is the aircraft. Geometry checks cannot certify
   object semantics, correct viewpoint, or source image licensing.
3. Decode the training frame to 8-bit grayscale and read its GT box. This initial
   implementation replaces one target per frame. Multi-target frames are retained
   unchanged, not partly relabeled.
4. Fit a robust local background plane outside an expanded GT box. Reject complex
   backgrounds and unreliable target contrast. Supported target dimensions are
   at least 3 px on the short edge and at most 160 px on the long edge, in the
   original frame, not the resized network input.
5. Search neighboring frames at offsets +/-10, 20, 40 and 80. Exclude every
   annotated donor target and frame borders. Use forward/backward-checked LK
   features and RANSAC to estimate camera motion, with masked ECC for low-texture
   sky. Apply the warp to the real donor image and correct low-frequency local
   brightness drift. Local background MAE must be <=3 gray levels and p95 error
   <=8. Source-vs-donor residuals define connected target/rotor/codec-halo masks;
   dilate and feather that contour, not an entire rectangular box. Require full
   valid donor coverage of the repair support and mean boundary change <=1.5
   gray levels. Preserve original background pixels outside this contour. Choose
   one suitable real donor per source frame, avoiding multi-frame texture averaging
   and independent random-noise fills. These checks are conservative heuristics,
   not proof that the optical scene or sensor noise is perfectly reproduced.
6. Match the original geometric long edge, preserve the new aircraft's aspect
   ratio, and place it at the original GT center. A 4x supersampled premultiplied
   alpha projection avoids transparent-color fringes. Match robust signed target
   contrast relative to local background, with bounded gain and clipping checks.
7. Apply an approximate 0.65 px Gaussian blur. `--blur-sigma` overrides it in
   original-frame pixels. This is not an estimated optical PSF, motion blur or a
   measured reproduction of the camera's codec/sharpening artifacts.
8. Recompute YOLO labels from `blurred_alpha > 0.15`. Label dimensions can differ
   from geometric dimensions due to the blur and the aircraft's silhouette.
   Do not interpret "same long edge" as "same width, height and occupied area".
9. Save lossless PNGs, labels, edit masks, metrics and previews. Reload PNGs and
   assert that every pixel outside the edit mask equals the decoded source gray
   pixel. Preview overlays are never burned into training images.

## Outputs

### Isolated Batch From the Current Training Pool

`build_gray_replacement_batch.py` consumes the `native_approved_expansion.v1`
dataset manifest and its exact `train_hardneg.txt`. It intersects approved video
SHA256 identities and actual scheduled image paths, rejects the held-out video
hashes, and excludes uncertain/unreviewed frames. Only approved single-target
frames supported by the temporal repair method are eligible. Older videos without
this approved registry are not silently used. All original training data remains
unchanged, including targets outside the synthesis size range.

```bash
python scripts/anti_uav/build_gray_replacement_batch.py \
  --dataset /mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_28videos_20260916 \
  --catalog /mnt/andrew/anti_uav_model_refinement/data/drone_asset_catalog_20260916/catalog.json \
  --output /mnt/andrew/anti_uav_model_refinement/data/real_gray_replacement_candidates_20260916 \
  --per-video 20 --variants 4 --previews 16 --exclude-assets 24,25
```

The fresh output contains successful replacements only, with lossless single-channel
PNGs, updated labels, edit masks, provenance, `accepted.jsonl`, `summary.json`, and
`train_synthetic.txt`. Rejected replacements are logged rather than copied into
the new image directory. Source selection balances videos and target short-edge
strata at a hypothetical 960x544 letterbox input; it is not a new training sampler.
Assets vary across frames, so these files are detector augmentation candidates,
not a temporally consistent tracker sequence. IDs 24 and 25 in this catalog are
excluded because their images include a remote control. Visual review and licensing
review are still required for the remaining assets.

The batch does not start training or append to any active list. It is positive-only:
review it first, then use a separate controlled experiment with an explicit positive
replacement fraction and the existing negative ratio. `status.json` reports running,
complete or failed; a failed job's partial files are not a completed dataset.

- `catalog/catalog.json`: metadata, raw picture provenance, cutout hashes/status.
- `catalog/raw/`: all extracted workbook pictures.
- `catalog/cutouts/`: usable native-alpha candidates only.
- `catalog/catalog_preview_*.jpg`: candidate sheets for visual review.
- `samples/images/`: full-resolution, single-channel, unannotated PNGs.
- `samples/labels/`: corresponding YOLO labels, empty for original negatives.
- `samples/masks/`: editable-region masks, all zero for unchanged samples.
- `samples/manifest.json`: per-sample provenance, asset assignment, metrics,
  skip reasons, timing and explicit limitations.
- `samples/previews/`: PNG original/replaced full frames, equal-scale pixel crops,
  source cutout and numerical checks. Green corners show original label boxes;
  cyan corners show the saved updated YOLO labels. Lines are drawn after resizing
  at 1 display pixel, with no cross or text over the target.
- `samples/preview_contact_sheet.jpg`: overview of the comparison pages.

To add bbox overlays to a previously generated run without modifying any training
image or label, use a new output directory:

```bash
python scripts/anti_uav/synthesize_gray_drone_replacements.py preview \
  --manifest deliverables/my_drone_synthesis/samples/manifest.json \
  --catalog deliverables/my_drone_synthesis/catalog/catalog.json \
  --output deliverables/my_drone_synthesis/previews_bbox --limit 10
```

The renderer reads the actual saved YOLO labels, verifies agreement with the
synthesis manifest and checks source/output/cutout hashes. The old box appears
only on the original panel; the new box only on the synthetic panel. Preview PNGs
and `preview_manifest.json` are separate from the lossless training images.

Compare a previous run against the temporal revision with identical source and
asset identities:

```bash
python scripts/anti_uav/compare_gray_background_repair.py \
  --old-manifest deliverables/old_run/manifest.json \
  --new-manifest deliverables/temporal_run/manifest.json \
  --output deliverables/background_comparison
```

This generates three-way original/old/new crops plus binary edit-mask comparisons.
The comparison includes no training modifications. New temporal metrics in the
synthesis manifest record the donor frame, approved hashes, affine transform,
registration evidence, background residuals, boundary change, contour support and
rejection reasons.

Unreliable positive replacements are saved as the unchanged original with their
original label. Negative frames remain unchanged. Each full variant preserves
the input frame count and positive/negative count. This does not automatically
enforce the training experiment's neg15 sampling schedule: use that experiment's
sampler and replace only a chosen fraction of positive slots after review.

## Limits and Experiment Use

This version outputs extracted frames, not a rendered continuous video stream.
It reads real neighboring video frames to repair the background. Asset identity
is consistent within each video/variant, but donor choices, photometric estimates
and registration are not yet constrained across a whole synthesized sequence.
It does not estimate 3D pose, preserve foreground occlusions, guarantee perfect
hidden-background recovery, or simulate rolling shutter/rotor motion. Warping
changes donor noise statistics; neighboring exposures and codec blocks can differ.
High-pass boundary statistics are reported as diagnostics, not a realism score.
These results must not be presented as real footage or as 334 newly captured
drone types. A single product view does not imply multiple real viewpoints.

Keep the current training run as the baseline. For a future controlled study,
replace e.g. 10% or 20% of positive sample slots while holding total sampling,
negative fraction, initialization and optimization fixed. Evaluate real held-out
videos; do not tune synthesis on Video00004. No performance gain is claimed by
this script.

## Tests

In the repository training environment:

```bash
python -m pytest -q tests/test_gray_drone_replacements.py
```

The repository's global test configuration imports the full YOLO stack. For a
standalone CPU-only environment without Torch, copy this test module into a
temporary non-package directory and run it with `PYTHONPATH` set to the repository
root and `pytest --noconftest`. The actual module under test remains the repository
script. Tests cover anchor-to-row mapping, matte checks, deterministic compositing,
geometry/contrast/background invariants, transparent-color halos, invalid labels,
holdout/path guards, negative preservation and saved output files. Temporal tests
also cover nonrectangular contour support, moving-target exclusion during camera
registration, actual background recovery on known synthetic fixtures, donor-target
overlap rejection, no-silent-fallback policy and held-out donor rejection.
