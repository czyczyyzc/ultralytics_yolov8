"""Offline audit of timestamped native camera runs; never part of inference."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text())
    with path.with_name("latency.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != data["frames"]:
        raise ValueError(f"{path}: incomplete timestamp log")
    periods, gaps = [], 0
    for index, row in enumerate(rows):
        if int(row["index"]) != index or not int(row["timestamp_valid"]):
            raise ValueError(f"{path}: invalid frame index/timestamp")
        values = [float(row[key]) for key in ("frame_ms", "dequeue_ms", "output_ms")]
        if not all(math.isfinite(value) for value in values) or values != sorted(values):
            raise ValueError(f"{path}: timestamp order/clock mismatch")
        if index:
            previous = rows[index - 1]
            delta = (int(row["sequence"]) - int(previous["sequence"])) % (2**32)
            if not 0 < delta < 2**31:
                raise ValueError(f"{path}: repeated or out-of-order sensor sequence")
            gaps += delta - 1
            periods.append((values[0] - float(previous["frame_ms"])) / delta)
    if gaps != data["camera"]["sequence_gaps"]:
        raise ValueError(f"{path}: gap count mismatch")
    measured = rows[data["args"]["warmup"] :]
    frame_age = statistics.mean(float(row["frame_to_output_ms"]) for row in measured)
    if abs(frame_age - data["stages_ms"]["driver_to_output"]["mean"]) > 1e-6:
        raise ValueError(f"{path}: latency summary mismatch")
    hardware = [data["hardware_before"], data["hardware_after"]]
    hardware += [sample["hardware"] for sample in data["hardware_samples"]]
    temperatures = [float(value) / 1000 for item in hardware
                    for key, value in item.items() if key.startswith("/sys/class/thermal/")]
    return {
        "frames": len(rows),
        "measured_seconds": data["measured_seconds"],
        "workers": data["args"]["workers"],
        "camera": data["camera"],
        "processed_fps": data["steady_fps"],
        "observed_sensor_fps": 1000 / statistics.median(periods) if periods else None,
        "skipped_fraction_between_first_and_last": gaps / (len(rows) + gaps),
        "first_result": data["first_result"],
        "stages_ms": data["stages_ms"],
        "detections": data["detector_count"],
        "displayed_tracks": data["displayed_tracks"],
        "peak_sampled_temperature_c": max(temperatures) if temperatures else None,
        "npu_warmup_per_worker": data["npu_warmup_per_worker"],
        "npu_warmup_ms": data["npu_warmup_ms"],
        "model_sha256": data["model_sha256"],
        "timestamp_flags": sorted({int(row["flags"]) for row in rows}),
        "audit": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = {path.parent.name: summarize(path) for path in sorted(args.directory.glob("*/summary.json"))}
    if not runs:
        raise ValueError("No completed camera runs")
    if len({run["model_sha256"] for run in runs.values()}) != 1:
        raise ValueError("Comparison mixes different models")
    result = {"runs": runs, "startup_medians": {}}
    for warm in (0, 3):
        selected = [run for name, run in runs.items() if name.startswith(f"startup_w{warm}_")]
        if selected:
            result["startup_medians"][str(warm)] = {
                "trials": len(selected),
                **{key: statistics.median(run["first_result"][key] for run in selected)
                   for key in selected[0]["first_result"]},
                "npu_warmup_ms": statistics.median(run["npu_warmup_ms"] for run in selected),
            }
    (args.directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, run in runs.items():
        if name.startswith("startup_"):
            continue
        stages = run["stages_ms"]
        print(f'{name}: {run["processed_fps"]:.2f} FPS, '
              f'read/result {stages["ordered_latency"]["mean"]:.2f} ms, '
              f'driver/result {stages["driver_to_output"]["mean"]:.2f} ms, '
              f'p95 {stages["driver_to_output"]["p95"]:.2f} ms, '
              f'skipped {100 * run["skipped_fraction_between_first_and_last"]:.1f}%')


if __name__ == "__main__":
    main()
