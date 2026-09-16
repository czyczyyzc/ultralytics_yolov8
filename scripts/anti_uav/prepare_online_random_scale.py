#!/usr/bin/env python3
"""Prepare all-donor online scale training without starting a training run."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from scripts.anti_uav.online_random_scale import eligible_donors, pixel_box, random_context_crop, gray_capture_jitter
from scripts.anti_uav.large_target_augmentation import yolo_rows


def preview_and_audit(donors, output, views, seed):
    image_dir = output/"preview"
    image_dir.mkdir()
    rng = np.random.default_rng(seed)
    reports, selected = [], {}
    targets = [.04, .17, .37, .65, .86, .97]
    for donor in donors:
        image = cv2.imread(donor["im_file"])
        if image is None:
            raise FileNotFoundError(donor["im_file"])
        for variant in range(views):
            result, box, meta = random_context_crop(image, pixel_box(donor), rng)
            reports.append(dict(source=donor["im_file"], variant=variant, **meta))
            for slot, target in enumerate(targets):
                distance = abs(meta["area_fraction"]-target)
                if slot not in selected or distance < selected[slot][0]:
                    selected[slot] = distance, result.copy(), box.copy(), reports[-1]
    canvas = np.full((660, 1440, 3), 25, dtype=np.uint8)
    for slot, (_, image, box, meta) in selected.items():
        path = image_dir/f"sample_{slot}.jpg"
        assert cv2.imwrite(str(path), image)
        path.with_suffix(".txt").write_text(yolo_rows([box]))
        small = cv2.resize(image, (480, 272))
        x1, y1, x2, y2 = np.round(box*.5).astype(int)
        cv2.rectangle(small, (max(0, x1), max(0, y1)), (min(479, x2), min(271, y2)), (70, 235, 70), 1)
        x, y = slot % 3*480, slot//3*330
        canvas[y+42:y+314, x:x+480] = small
        cv2.putText(canvas, f'BBox {meta["area_fraction"]*100:.1f}% | resize {meta["upscale"]:.2f}x',
                    (x+8, y+19), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'ONLINE RANDOM SCALE | green: updated GT', (x+8, y+36),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (180, 180, 180), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(output/"scale_preview.jpg"), canvas)
    (output/"preview_sources.json").write_text(json.dumps([r[3] for r in selected.values()], indent=2)+"\n")
    (output/"preview_draws.json").write_text(json.dumps(reports, indent=2)+"\n")
    counts, _ = np.histogram([r["area_fraction"] for r in reports], [0, .02, .1, .25, .5, .8, 1.001])
    return dict(sampled_draws=len(reports), covered_donors=len({r["source"] for r in reports}),
                area_bins=[0, .02, .1, .25, .5, .8, 1.001], sampled_area_counts=counts.tolist(),
                partial_count=sum(r["partial"] for r in reports),
                partial_requested=sum(r["requested_partial"] for r in reports),
                max_resize=max(r["upscale"] for r in reports), min_retained=min(r["retained_fraction"] for r in reports),
                seed=seed, note="QA draws only. Training re-samples, not a saved augmentation training set.")


def smoke_loader(data_path):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from scripts.anti_uav.online_random_scale import OnlineScaleDataset
    torch.set_num_threads(4)
    np.random.seed(20260916)
    data = yaml.safe_load(data_path.read_text())
    data["nc"] = 1
    cfg = get_cfg(overrides=dict(imgsz=[544, 960], mosaic=0., mixup=0., copy_paste=0., scale=.2, translate=.05,
                                fliplr=.5, flipud=0., cache=False, rect=False, single_cls=True))
    base = build_yolo_dataset(cfg, data["train"], 4, data, mode="train")
    wrapped = OnlineScaleDataset(base, data["online_scale"])
    # Explicitly mix native positives, negatives, and virtual entries in the same batches.
    native_positive = next(i for i, r in enumerate(base.labels) if len(r["bboxes"]))
    native_negative = next(i for i, r in enumerate(base.labels) if not len(r["bboxes"]))
    indices = [native_positive, native_negative, len(base), len(wrapped)-1]*2
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(wrapped, indices), batch_size=4,
                                         num_workers=2, collate_fn=wrapped.collate_fn,
                                         worker_init_fn=__import__("ultralytics.data.build", fromlist=["seed_worker"]).seed_worker)
    shapes = []
    for batch in loader:
        assert tuple(batch["img"].shape) == (4, 3, 544, 960)
        assert torch.isfinite(batch["bboxes"]).all() and ((batch["bboxes"] >= 0) & (batch["bboxes"] <= 1)).all()
        assert len(batch["bboxes"]) == len(batch["cls"]) == len(batch["batch_idx"])
        shapes.append(list(batch["img"].shape))
    # Resampling is deterministic under a reset seed, but different on successive visits.
    np.random.seed(17)
    first = wrapped[len(base)]
    second = wrapped[len(base)]
    np.random.seed(17)
    replay = wrapped[len(base)]
    assert torch.equal(first["img"], replay["img"]) and torch.equal(first["bboxes"], replay["bboxes"])
    assert not torch.equal(first["img"], second["img"])
    return dict(mixed_batch_shapes=shapes, base_slots=len(base), total_slots=len(wrapped),
                seed_reproducible=True, resampled_on_repeat=True, workers=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--views-per-donor", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--smoke-loader", action="store_true")
    args = p.parse_args()
    if args.views_per_donor < 1 or args.output.exists():
        raise ValueError("Use a fresh output and at least one view per donor")
    cv2.setNumThreads(1)
    source = json.loads((args.source/"manifest.json").read_text())
    zoom = json.loads((args.source/"zoom_manifest.json").read_text())
    removed = {r["image"] for r in zoom["train"]}
    original = (args.source/"train_hardneg.txt").read_text().splitlines()
    base = [x for x in original if x not in removed]
    assert len(original)-len(base) == len(removed) and len(base) == source["append_only_entries"]
    cache = np.load(args.source/"train_hardneg.cache", allow_pickle=True).item()
    assert Counter(r["im_file"] for r in cache["labels"]) == Counter(original)
    records = {r["im_file"]: r for r in cache["labels"]}
    donors = eligible_donors([records[x] for x in sorted(set(base))])
    assert donors
    val_paths = (args.source/"val_monitor.txt").read_text().splitlines()
    assert not set(base) & set(val_paths)
    val_hash = source["validation_video"]["sha256"][:12]
    for path in base:
        assert "Video00004" not in Path(path).parts and val_hash not in path
    args.output.mkdir(parents=True)
    (args.output/"train_hardneg.txt").write_text("\n".join(base)+"\n")
    (args.output/"val_monitor.txt").write_text("\n".join(val_paths)+"\n")
    donor_manifest = args.output/"online_scale_donors.json"
    rows = [{"im_file": r["im_file"], "shape": list(r["shape"]), "bboxes": np.asarray(r["bboxes"]).tolist()} for r in donors]
    donor_manifest.write_text(json.dumps(dict(donors=rows), indent=2)+"\n")
    data = dict(path=str(args.output), train=str(args.output/"train_hardneg.txt"), val=str(args.output/"val_monitor.txt"),
                names={0: "drone"}, online_scale=dict(donor_manifest=str(donor_manifest), views_per_donor=args.views_per_donor,
                                                    max_upscale=4., partial_probability=.15))
    data_path = args.output/"train_hardneg_gray_monitor.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False))
    final = len(base)+len(donors)*args.views_per_donor
    manifest = dict(source, schema="online_random_scale.v1", source_dataset=str(args.source),
                    static_zoom_removed=len(removed), zoom_training_samples=0,
                    online_donors=len(donors), online_views_per_donor=args.views_per_donor,
                    online_additional_slots=len(donors)*args.views_per_donor, final_entries=final,
                    final_negative_fraction=source["negative"]/final,
                    all_native_positive_and_negative_exposure_preserved=True,
                    augmentation_type="Fresh random context crops per visit; base schedule unchanged. No automatic training launch.")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    report = preview_and_audit(donors, args.output, args.views_per_donor, args.seed)
    report.update(native_slots=len(base), total_slots=final, eligible_donors=len(donors), views_per_donor=args.views_per_donor)
    if args.smoke_loader:
        report["loader_smoke"] = smoke_loader(data_path)
    (args.output/"online_scale_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
