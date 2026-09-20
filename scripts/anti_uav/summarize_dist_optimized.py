#!/usr/bin/env python3
"""Summarize measured optimized runs and verify their detector observations."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    baseline = read(args.baseline / "summary.json")
    reference = [json.loads(line) for line in (args.baseline / "observations.jsonl").read_text().splitlines()]
    runs = {}
    for path in sorted(args.root.glob("*/summary.json")):
        report = read(path)
        if "steady_fps" not in report or "npu_core_masks" not in report:
            continue
        assert report["model_sha256"] == baseline["model_sha256"]
        assert report["args"]["conf"] == .03 and report["args"]["iou"] == .45
        observations = path.parent / "observations.jsonl"
        identical_tracks = checked = 0
        if observations.is_file():
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            assert len(rows) == report["frames"] <= len(reference)
            for index, row in enumerate(rows):
                assert row["frame_index"] == index
                assert row["boxes_xyxy_score"] == reference[index]["boxes_xyxy_score"], (path,index)
                identical_tracks += row["displayed_tracks"] == reference[index]["displayed_tracks"]
            checked = len(rows)
        all_seconds = report["frames"] / report["all_frames_fps"]
        previous = next((sample for sample in report["hardware_samples"] if sample["frame"] == 12000), None)
        last_fps = ((report["frames"]-12000) / (all_seconds-previous["elapsed_seconds"])) if previous and report["frames"] > 12000 else None
        samples = [report["hardware_before"], *[sample["hardware"] for sample in report["hardware_samples"]], report["hardware_after"]]
        npu_key = "/sys/class/devfreq/fdab0000.npu/cur_freq"
        runs[path.parent.name] = dict(frames=report["frames"], fps=report["steady_fps"],
            last_2201_fps=last_fps, stages_ms=report["stages_ms"], first_result=report["first_result"],
            bit_exact_detector_frames=checked, identical_tracker_frames=identical_tracks if checked else None,
            npu_hz_samples=sorted({int(sample[npu_key]) for sample in samples}),
            maximum_soc_c=max(int(sample["/sys/class/thermal/thermal_zone0/temp"])/1000 for sample in samples),
            npu_worker_frame_counts=report.get("npu_worker_frame_counts"),
            args=report["args"])
    result = dict(model_sha256=baseline["model_sha256"], baseline_fps=baseline["steady_fps"], runs=runs,
        quality={p.parent.name:read(p) for p in sorted(args.root.glob("quality_*/summary.json"))},
        scope="Measured live RKNN video processing; no rendering/encoding. Runs have different thermal histories, not a controlled isolated cooling experiment. GMC replay times are not board FPS. Video00009 is diagnostic/tuning data, not independent generalization evidence.")
    startup = args.root / "startup_final/summary.json"
    if startup.is_file():
        result["startup"] = read(startup)
    (args.root / "comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    for name, run in runs.items():
        print(name, f"{run['fps']:.2f} FPS", "last2201", run["last_2201_fps"], "exact", run["bit_exact_detector_frames"])


if __name__ == "__main__":
    main()
