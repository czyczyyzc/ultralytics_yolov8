#!/usr/bin/env python3
"""Wait for the 44-video cache, enable all screened cutouts, then train and compare."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--asset-cache", type=Path, required=True)
    p.add_argument("--candidate-catalog", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--initial-p3", type=Path, required=True)
    p.add_argument("--bundle", type=Path, help="Exact pushed commit bundle to pull after cache preparation")
    p.add_argument("--device", type=int, default=2)
    p.add_argument("--timeout-hours", type=float, default=48)
    a = p.parse_args()
    for key in ("repo", "source_cache", "asset_cache", "candidate_catalog",
                "run_dir", "reference_run", "initial_p3"):
        setattr(a, key, getattr(a, key).resolve())
    if a.bundle is not None:
        a.bundle = a.bundle.resolve()
    if a.timeout_hours <= 0 or a.run_dir.exists() or a.asset_cache.exists():
        raise ValueError("Use fresh output paths and a positive timeout")
    a.run_dir.mkdir(parents=True)
    status_path = a.run_dir / "orchestration_status.json"

    def status(stage: str, **extra) -> None:
        record = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(record), flush=True)

    def execute(script: str, args: list[str], log_name: str) -> None:
        command = [sys.executable, str(a.repo / "scripts/anti_uav" / script), *args]
        with (a.run_dir / log_name).open("x") as stream:
            subprocess.run(command, cwd=a.repo, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        status("waiting_for_backgrounds")
        deadline = time.monotonic() + a.timeout_hours * 3600
        while True:
            state_path = a.source_cache / "extension_status.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                if state["stage"] == "complete":
                    break
                if state["stage"] == "failed":
                    raise RuntimeError(f"Background preparation failed: {state.get('error')}")
                os.kill(int(state["pid"]), 0)
            if time.monotonic() >= deadline:
                raise TimeoutError("Background preparation exceeded timeout")
            time.sleep(60)
        summary = json.loads((a.source_cache / "summary.json").read_text())
        if summary["stage"] != "complete" or summary["source_videos"] != 44 or summary["asset_count"] != 53:
            raise ValueError("Expected a complete 44-video, 53-asset background cache")
        if a.bundle is not None:
            status("syncing_code")
            subprocess.run(["git", "pull", "--ff-only", str(a.bundle), "main"], cwd=a.repo, check=True)
        status("expanding_assets", ready_backgrounds=summary["ready_backgrounds"])
        execute("prepare_gray_asset_ablation.py", [
            "--source", str(a.source_cache), "--candidates", str(a.candidate_catalog),
            "--output", str(a.asset_cache),
        ], "expand_assets.log")
        assets = json.loads((a.asset_cache / "summary.json").read_text())
        if assets["stage"] != "complete" or assets["asset_count"] != 328:
            raise ValueError("Expanded cache does not contain all 328 screened assets")
        status("training_and_comparison", assets=assets["asset_count"])
        execute("run_online_gray_expansion_training.py", [
            "--cache", str(a.asset_cache), "--run-dir", str(a.run_dir / "experiment"),
            "--initial-p3", str(a.initial_p3), "--reference-run", str(a.reference_run),
            "--old-run", str(a.reference_run), "--device", str(a.device),
            "--epochs", "15", "--comparison-kind", "combined",
            "--reference-label", "approved40", "--experiment-label", "approved44",
            "--preview-count", "20",
        ], "training_and_comparison.log")
        status("complete", report=str(a.run_dir / "experiment/COMPARISON.md"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
