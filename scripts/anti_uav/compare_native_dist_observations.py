#!/usr/bin/env python3
"""Offline exact comparison of recorded pipelines, never part of inference."""
import argparse
import json
from pathlib import Path


def compare(reference, candidate, prefix=None):
    if prefix is not None and prefix < 1:
        raise ValueError("prefix must be positive")
    keys = ("boxes_xyxy_score", "displayed_tracks", "warp")
    differences = {key: [] for key in keys}
    counts = {key: 0 for key in keys}
    frames = observations = 0
    while prefix is None or frames < prefix:
        left, right = reference.readline(), candidate.readline()
        if not left or not right:
            if bool(left) != bool(right) or prefix is not None:
                raise ValueError(f"Frame count mismatch at {frames}")
            break
        a, b = json.loads(left), json.loads(right)
        if a["frame_index"] != frames or b["frame_index"] != frames:
            raise ValueError(f"Missing or unordered frame at {frames}")
        for key in keys:
            if a[key] != b[key]:
                counts[key] += 1
                if len(differences[key]) < 20:
                    differences[key].append(frames)
        observations += len(b["displayed_tracks"])
        frames += 1
    if not frames:
        raise ValueError("Empty observations")
    if prefix is not None and candidate.readline():
        raise ValueError("Candidate exceeds requested prefix")
    return dict(passed=not any(counts.values()), frames=frames, observations=observations,
                differing_frames=counts, first_differing_frames=differences,
                scope="Exact recorded detection/ID/current-box/warp comparison; no timing or FPS measurement.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=int)
    args = parser.parse_args()
    with args.reference.open() as a, args.candidate.open() as b:
        report = compare(a, b, args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
