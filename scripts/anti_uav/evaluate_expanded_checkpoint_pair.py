#!/usr/bin/env python3
"""Compare fixed old/new checkpoints on identical native validation and test frames."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--validation-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    import numpy as np
    import torch
    import yaml
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator, pool_native
    from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
    from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False,
                         hub=False, mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = json.loads(a.models.read_text())
    splits = {"Video00004_test": (HOLDOUT, (2359, 448)),
              "Video00009_selection_subset": (a.validation_data, (1421, 797))}
    datasets = {}
    for name, (config, expected) in splits.items():
        listing = Path(yaml.safe_load(config.read_text())["val"])
        paths = [v for v in listing.read_text().splitlines() if v and "/zoom_val/" not in v]
        if len(paths) != expected[0] or len(set(paths)) != len(paths):
            raise ValueError(f"Native split changed: {name}")
        datasets[name] = dict(config=str(config), listing=str(listing), sha256=sha256(listing))
    results = {}
    for name, checkpoint in models.items():
        entries, by_split = [], {}
        digest = sha256(checkpoint)
        for split, (config, expected) in splits.items():
            dump(a.output / "status.json", dict(stage="evaluating", model=name, split=split))
            model = YOLO(checkpoint)
            metrics = model.val(data=str(config), validator=PairValidator, imgsz=[544, 960], rect=False,
                device=a.device, batch=32, workers=0, conf=.001, iou=.45, max_det=100,
                half=False, plots=False, verbose=False, project=str(a.output / "validation"), name=name+"_"+split)
            native = {k: v for k, v in clean(metrics.gray_selection).items() if k.startswith("native/")}
            if (native["native/c0.03/FRAMES"], native["native/c0.03/TP"] + native["native/c0.03/FN"]) != expected:
                raise ValueError(f"Frame/GT coverage mismatch: {name}/{split}")
            by_split[split] = native
            entries.append(dict(metrics=native, arrays=metrics.native_pr_arrays, scales=metrics.native_scale_counts))
            np.savez_compressed(a.output / f"{name}_{split}_pr.npz", **metrics.native_pr_arrays)
            del model
            torch.cuda.empty_cache()
        pooled, arrays = pool_native(entries)
        by_split["pooled_native"] = clean(pooled)
        np.savez_compressed(a.output / f"{name}_pooled_pr.npz", **arrays)
        if sha256(checkpoint) != digest:
            raise ValueError("Checkpoint changed during comparison")
        results[name] = dict(weights=checkpoint, sha256=digest, splits=by_split)
        dump(a.output / "results.json", results)
    for record in datasets.values():
        if sha256(record["listing"]) != record["sha256"]:
            raise ValueError("Evaluation labels/list changed")
    dump(a.output / "protocol.json", dict(datasets=datasets, input_hw=[544, 960],
        precision="FP32", conf_floor=.001, confidence=.03, nms_iou=.45, match_iou=.5,
        pooled_frames=3780, pooled_gt=1245, independent_pooled_test=False,
        note="Video00009 selects checkpoints; pooled AP globally recomputed; no tracker."))
    dump(a.output / "status.json", dict(stage="complete", models=len(models)))


if __name__ == "__main__":
    main()
