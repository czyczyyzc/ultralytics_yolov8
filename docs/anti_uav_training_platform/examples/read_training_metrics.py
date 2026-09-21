#!/usr/bin/env python3
"""Read completed epoch rows without importing Torch or changing training files."""

import argparse
import csv
import io
import json
import math
from pathlib import Path


STAGES = (("p3", "training_p3/p3/results.csv"),
          ("addon_p2", "training_addon/p2/results.csv"))


def read_points(run_dir, epochs_per_stage, after_sequence=0):
    points = []
    for stage_index, (stage, relative) in enumerate(STAGES):
        path = Path(run_dir) / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        # The trainer appends one row per completed epoch. Ignore a partial tail.
        end = text.rfind("\n")
        if end < 0:
            continue
        rows = csv.reader(io.StringIO(text[:end + 1]))
        header = [key.strip() for key in next(rows, [])]
        if not header or "epoch" not in header:
            continue
        stage_points = {}
        for row in rows:
            if len(row) != len(header):
                continue
            try:
                raw = dict(zip(header, (float(value.strip()) for value in row)))
            except ValueError:
                continue
            epoch = raw.pop("epoch")
            if not math.isfinite(epoch) or not epoch.is_integer() or not 1 <= epoch <= epochs_per_stage:
                continue
            epoch = int(epoch)
            values = {key: value if math.isfinite(value) else None for key, value in raw.items()}
            fields = ("native/c0.03/F2", "native/mAP50", "native/mAP50-95")
            if all(values.get(key) is not None for key in fields):
                values["selection/fitness_reconstructed"] = sum(
                    weight * values[key] for weight, key in zip((.5, .3, .2), fields)
                )
            sequence = stage_index * epochs_per_stage + epoch
            stage_points[epoch] = dict(sequence=sequence, stage=stage, epoch=epoch,
                                       global_epoch=sequence, source="results.csv", values=values)
        points.extend(stage_points.values())
    points.sort(key=lambda point: point["sequence"])
    return dict(items=[point for point in points if point["sequence"] > after_sequence],
                latest_sequence=max((point["sequence"] for point in points), default=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs-per-stage", type=int, required=True)
    parser.add_argument("--after-sequence", type=int, default=0)
    args = parser.parse_args()
    if args.epochs_per_stage < 1 or args.after_sequence < 0:
        parser.error("Epoch count must be positive and cursor nonnegative")
    print(json.dumps(read_points(args.run_dir, args.epochs_per_stage, args.after_sequence),
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
