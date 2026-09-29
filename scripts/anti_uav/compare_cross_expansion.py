#!/usr/bin/env python3
"""Compare 40- and 41-video P3/P2 detectors on unchanged gray evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
from scripts.anti_uav.synthesize_gray_drone_replacements import sha256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--new-run", type=Path, required=True)
    p.add_argument("--data-yaml", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="1")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    models = {
        "old_p3": a.baseline_run / "training_p3/p3/weights/best.pt",
        "old_p2p3": a.baseline_run / "training_addon/p2/weights/best.pt",
        "new_p3": a.new_run / "training_p3/p3/weights/best.pt",
        "new_p2p3": a.new_run / "training_addon/p2/weights/best.pt",
    }
    results = {}
    for split, data in (("Video00009_selection_subset", a.data_yaml),
                        ("Video00004_test", HOLDOUT)):
        for name, weights in models.items():
            if not weights.is_file():
                raise FileNotFoundError(weights)
            metrics = YOLO(str(weights)).val(
                data=str(data), imgsz=[544, 960], rect=False,
                validator=FixedShapeGrayValidator, device=a.device, batch=32,
                workers=4, conf=.001, iou=.45, max_det=100, half=False,
                plots=False, project=str(a.output / "val"),
                name=f"{split}_{name}", verbose=False)
            record = dict(weights=str(weights), sha256=sha256(weights),
                          metrics=clean(metrics.gray_selection))
            results[f"{split}_{name}"] = record
            (a.output / f"{split}_{name}.json").write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n")
    (a.output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n")
    lines = ["# Reviewed Cross-Target Video Expansion", "",
             "FP32 detection at 960x544; no RKNN or tracking.",
             "The 40-video training schedule is preserved and one reviewed video is added.",
             "The _x video was already in the 40-video baseline. Neither cross video is an independent cross-target test.",
             "Video00009 selects checkpoints; Video00004 is test-only.", "",
             "| Split | Model | Conf | Precision | Recall | FP | mAP50 | 4-8px recall |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("Video00009_selection_subset", "Video00004_test"):
        for name in models:
            m = results[f"{split}_{name}"]["metrics"]
            for conf in (.01, .03, .05):
                prefix = f"native/c{conf:.2f}"
                tiny = m[prefix + "/long_4to8px/R"]
                tiny_text = "N/A" if tiny is None else f"{tiny:.2%}"
                lines.append(f"| {split} | {name} | {conf:.2f} | {m[prefix+'/P']:.2%} | "
                             f"{m[prefix+'/R']:.2%} | {m[prefix+'/FP']} | "
                             f"{m['native/mAP50']:.2%} | {tiny_text} |")
    (a.output / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print(a.output / "COMPARISON.md", flush=True)


if __name__ == "__main__":
    main()
