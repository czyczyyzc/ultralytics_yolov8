#!/usr/bin/env python3
"""Check complete board outputs and summarize the measured deployment runs."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def frames(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row["frame_index"] == i for i, row in enumerate(rows)), path
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    names = ["final_detector_1", "final_detector_3", "final_pipeline_three_full"]
    reports, rows = {}, {}
    for name in names:
        folder = args.root / name
        reports[name] = read(folder / "summary.json")
        rows[name] = frames(folder / "observations.jsonl")
        report = reports[name]
        assert len(rows[name]) == report["frames"]
        assert report["padding_value"] == 114 and report["args"]["preload"] == 0
        assert report["args"]["conf"] == .03 and report["args"]["iou"] == .45
    assert len(set(report["model_sha256"] for report in reports.values())) == 1
    one, three, tracked = (rows[name] for name in names)
    assert len(one) == len(three)
    for a, b, c in zip(one, three, tracked):
        assert a["boxes_xyxy_score"] == b["boxes_xyxy_score"] == c["boxes_xyxy_score"], a["frame_index"]
    assert len(tracked) == 14201
    ids = set()
    for row in tracked:
        seen = set()
        for track in row["displayed_tracks"]:
            index = track["detection_index"]
            assert index not in seen
            seen.add(index)
            detection = row["boxes_xyxy_score"][index]
            assert track["box"] == detection[:4] and track["score"] == detection[4]
            ids.add(track["id"])
    selected = ("frames", "measured_frames", "steady_fps", "all_frames_fps",
                "first_result_from_python_entry_ms", "first_frame_read_to_result_ms",
                "model_load_ms", "stages_ms", "hardware_before", "hardware_after")
    result = dict(passed=True, full_frames=len(tracked),
        bit_exact_detector_comparison_frames=len(one),
        confirmed_tracks_use_current_detector_boxes=True,
        full_detection_count=sum(len(r["boxes_xyxy_score"]) for r in tracked),
        full_displayed_track_count=sum(len(r["displayed_tracks"]) for r in tracked),
        distinct_visible_ids=len(ids), maximum_visible_id=max(ids, default=0),
        model_sha256=reports[names[0]]["model_sha256"],
        runs={name:{key:report[key] for key in selected} for name, report in reports.items()})
    startup = args.root / "startup_process_launch/summary.json"
    if startup.is_file():
        result["startup_process_launch"] = read(startup)
    (args.root / "benchmark_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("runs", "startup_process_launch")}))
    for name, report in reports.items():
        print(name, report["steady_fps"], "FPS")


if __name__ == "__main__":
    main()
