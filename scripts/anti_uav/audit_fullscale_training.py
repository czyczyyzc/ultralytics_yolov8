#!/usr/bin/env python3
"""Verify schedule preservation and render labeled context-zoom / hard-negative QA."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def zoom_preview(records, output):
    canvas = np.full((660, 1440, 3), 25, dtype=np.uint8)
    selected = []
    for i, target in enumerate((.075, .175, .375, .625, .87, 1.)):
        row = min(records, key=lambda r: abs(r["area_fraction"]-target))
        image = cv2.imread(row["image"])
        assert image is not None
        image = cv2.resize(image, (480, 272))
        x1, y1, x2, y2 = np.round(np.array(row["output_box"])*.5).astype(int)
        cv2.rectangle(image, (max(0, x1), max(0, y1)), (min(479, x2), min(271, y2)), (70, 235, 70), 1)
        x, y = (i % 3)*480, (i//3)*330
        canvas[y+42:y+314, x:x+480] = image
        title = f'BBox {row["area_fraction"]*100:.1f}% | crop resize {row["upscale"]:.2f}x'
        cv2.putText(canvas, title, (x+8, y+19), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'TRAIN AUGMENTATION | green: updated GT', (x+8, y+36), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (180, 180, 180), 1, cv2.LINE_AA)
        selected.append(row)
    assert cv2.imwrite(str(output/"scale_preview.jpg"), canvas)
    (output/"preview_sources.json").write_text(json.dumps(selected, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset/"manifest.json").read_text())
    zoom = json.loads((args.dataset/"zoom_manifest.json").read_text())
    old_data = yaml.safe_load(Path(manifest["old_data"]).read_text())
    old = Path(old_data["train"]).read_text().splitlines()
    base = (args.dataset/"train_append_only.txt").read_text().splitlines()
    final = (args.dataset/"train_hardneg.txt").read_text().splitlines()
    positives = set(manifest["source_training_positive_paths"])
    assert base[:len(old)] == old
    assert Counter(p for p in final if p in positives) == Counter(p for p in base if p in positives)
    old_cache = np.load(Path(old_data["train"]).with_suffix(".cache"), allow_pickle=True).item()
    old_positive = Counter(r["im_file"] for r in old_cache["labels"] if len(r["bboxes"]))
    final_counts = Counter(final)
    assert all(final_counts[path] == count for path, count in old_positive.items())
    small_entries = 0
    for row in old_cache["labels"]:
        h, w = row["shape"]
        wh = np.asarray(row["bboxes"]).reshape(-1, 4)[:, 2:]*[w, h]
        long_edge = wh.max(axis=1)*min(960/w, 544/h)
        small_entries += int(((long_edge >= 4) & (long_edge <= 8)).sum())
    assert {r["source"] for r in zoom["train"]} <= set(base)
    assert not {r["source"] for r in zoom["validation"]} & set(base)
    area = np.array([r["area_fraction"] for r in zoom["train"]])
    bins = [0., .05, .1, .25, .5, .8, .99, 1.001]
    counts, _ = np.histogram(area, bins=bins)
    audit = dict(old_schedule_entries=len(old), old_prefix_preserved=True,
                 old_positive_entries_preserved=sum(old_positive.values()),
                 old_4to8px_box_entries_preserved=small_entries,
                 native_train_entries=len(base), augmented_train_entries=len(final),
                 train_zoom_samples=len(area), train_zoom_at_least_80pct=int((area >= .8).sum()),
                 train_zoom_area_bins=bins, train_zoom_area_counts=counts.tolist(),
                 train_zoom_unique_sources=len({r["source"] for r in zoom["train"]}),
                 maximum_source_reuse=max(Counter(r["source"] for r in zoom["train"]).values()),
                 maximum_upscale=max(r["upscale"] for r in zoom["train"]),
                 area_definition="GT bounding-box area / image area, not foreground silhouette area",
                 limitation="Synthetic context crops include truncated close-ups; not proof of real close-range recall.")
    zoom_preview(zoom["train"], args.output)
    hard = json.loads((args.dataset/"hard_negative_manifest.json").read_text())
    audit["negative_replacements"] = hard["replacements"]
    audit["negative_entries"] = manifest["negative"]
    audit["negative_fraction"] = manifest["negative"]/len(final)
    changes = list({c["added"]: c for c in hard["changes"]}.values())
    changes.sort(key=lambda c: -c["score"])
    if changes:
        canvas = np.full((620, 1440, 3), 25, dtype=np.uint8)
        for i, row in enumerate(changes[:6]):
            image = cv2.imread(row["added"])
            assert image is not None
            image = cv2.resize(image, (480, 270))
            x, y = (i % 3)*480, (i//3)*310
            canvas[y+30:y+300, x:x+480] = image
            cv2.putText(canvas, f'Empty training GT | teacher conf {row["score"]:.3f}', (x+8, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
        assert cv2.imwrite(str(args.output/"hard_negative_preview.jpg"), canvas)
    (args.output/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
