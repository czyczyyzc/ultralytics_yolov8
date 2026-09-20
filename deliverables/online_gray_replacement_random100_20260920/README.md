# Random Online Replacement Review

Generated on 2026-09-20 from the completed 37-video background cache. Seed: 20260920.
100 unique frames were sampled randomly from the 49,002 cache-ready positive frames.
The 100 successful replacements cover 31 source videos; no attempted pair failed.
Cutouts use balanced shuffled cycles to cover every one of the 53 currently enabled IDs.
This is a visualization batch, not an added training dataset or a model-accuracy result.

Open `index.html` to browse all 100 comparisons. `contact_sheets/` contains ten pages,
ten comparisons per page. Individual 1200x780 cards are in `comparisons/`.
Green corners mark original boxes; cyan corners mark recomputed replacement boxes.
Unannotated full-resolution images are in `originals/` and `replacements/`, with
original and updated YOLO boxes in `original_labels/` and `labels/`.

Pixel checks confirm that nothing outside the foreground repair mask changed.
Images use lossless PNG storage; the gallery overview pages are JPEGs for browsing.
The source files, live training list and validation/test data were not modified.
Visual realism still requires human review; a successful integrity check is not
evidence that a pasted model has a realistic flight pose or will improve accuracy.

## Cutout Inventory Clarification

The extracted workbook catalog has 334 entries but only 323 distinct embedded image hashes.
55 entries have native alpha. Two controller-containing entries, IDs 24 and 25, are excluded.
The enabled 53 entries contain 50 distinct cutout images: IDs 1/34, 17/36, and 44/45
are duplicate pairs. Thus "53 available" means enabled catalog entries, not 53 unique silhouettes.
The remaining 279 entries are opaque and need foreground segmentation and quality review;
they have not been rejected as permanently unusable. Do not paste their rectangular backgrounds.

Expansion should segment the other images, remove accessories/shadows/text, check fine
rotor/wing structures and flight-view suitability, and deduplicate before publishing a new
versioned catalog. Existing frozen training/cache protocols must not be silently changed.
Reliable cached background repair is conceptually reusable with new cutouts, but the
current loader intentionally checks catalog hashes and IDs and requires an explicit new version.

Server export:
`/mnt/andrew/anti_uav_model_refinement/data/online_gray_replacement_random100_20260920/`

Generator:
`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/preview_online_gray_replacements.py`
