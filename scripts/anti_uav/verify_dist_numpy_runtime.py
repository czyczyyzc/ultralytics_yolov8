#!/usr/bin/env python3
"""Verify the no-PyTorch adapter against an existing full public-code replay."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from dist_numpy_runtime import CONFIG, as_results, load_dist, observations, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "detections", "expected", "gmc", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    expected_summary = json.loads((args.expected.parent / "summary.json").read_text())
    hashes = source_hashes(args.upstream)
    for path, digest in expected_summary["provenance"]["upstream_files"].items():
        if path.endswith(".py") and hashes[path] != digest:
            raise AssertionError(f"Upstream changed: {path}")
    assert expected_summary["config"] == CONFIG
    frames = [json.loads(line) for line in args.detections.read_text().splitlines()]
    expected = [json.loads(line) for line in args.expected.read_text().splitlines()]
    with np.load(args.gmc, allow_pickle=False) as data:
        warps = data["warps"]
    assert len(frames) == len(expected) == len(warps)
    tracker = load_dist(args.upstream)(SimpleNamespace(**CONFIG), frame_rate=100)

    class ReplayGMC:
        index = 0
        def apply(self, image, detections=None):
            warp = warps[self.index].copy()
            self.index += 1
            return warp

    tracker.gmc = ReplayGMC()
    count = 0
    for index, (frame, old) in enumerate(zip(frames, expected)):
        assert frame["frame_index"] == old["frame_index"] == index
        boxes = np.asarray(frame["boxes_xyxy_score"], np.float32).reshape(-1,5)
        tracker.update(as_results(boxes), img=np.empty((1,1,3),np.uint8))
        actual = observations(tracker, boxes)
        wanted = old["displayed_tracks"]
        assert len(actual) == len(wanted), index
        for a, b in zip(actual, wanted):
            assert a["id"] == b["id"] and a["detection_index"] == b["detection_index"], index
            np.testing.assert_allclose(a["box"], b["box"], rtol=1e-6, atol=1e-4)
            assert abs(a["score"]-b["score"]) < 1e-6, index
        count += len(actual)
    result = dict(passed=True, frames=len(frames), displayed_observations=count,
                  exact_ids_and_associations=True, config=CONFIG, source_hashes=hashes,
                  scope="Adapter equivalence with fixed observations/warps; NOT a speed benchmark")
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "source_hashes"}))


if __name__ == "__main__":
    main()
