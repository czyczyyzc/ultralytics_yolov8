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
5. Approximate the hidden sky with the fitted plane plus sampled local residual
   noise and an edge blend. Expand the erase region to include rotor/codec halos.
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
   pixel. JPEG preview pages are not the training images.

## Outputs

- `catalog/catalog.json`: metadata, raw picture provenance, cutout hashes/status.
- `catalog/raw/`: all extracted workbook pictures.
- `catalog/cutouts/`: usable native-alpha candidates only.
- `catalog/catalog_preview_*.jpg`: candidate sheets for visual review.
- `samples/images/`: full-resolution, single-channel, unannotated PNGs.
- `samples/labels/`: corresponding YOLO labels, empty for original negatives.
- `samples/masks/`: editable-region masks, all zero for unchanged samples.
- `samples/manifest.json`: per-sample provenance, asset assignment, metrics,
  skip reasons, timing and explicit limitations.
- `samples/previews/`: original/replaced full frames, equal-scale pixel crops,
  source cutout and numerical checks. No GT boxes, prediction boxes or crosses.
- `samples/preview_contact_sheet.jpg`: overview of the comparison pages.

Unreliable positive replacements are saved as the unchanged original with their
original label. Negative frames remain unchanged. Each full variant preserves
the input frame count and positive/negative count. This does not automatically
enforce the training experiment's neg15 sampling schedule: use that experiment's
sampler and replace only a chosen fraction of positive slots after review.

## Limits and Experiment Use

This version operates on extracted frames, not a video stream. Asset identity is
consistent within a video, but per-frame photometric estimates and sampled noise
are not temporally smoothed. It does not estimate 3D pose, preserve occlusions,
recover the true hidden background, or simulate rolling shutter/rotor motion.
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
holdout/path guards, negative preservation and saved output files.
