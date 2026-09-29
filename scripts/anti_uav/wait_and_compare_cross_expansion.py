#!/usr/bin/env python3
"""Run the fixed gray comparison once a two-stage cross expansion finishes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--new-run", type=Path, required=True)
    p.add_argument("--data-yaml", type=Path, required=True)
    p.add_argument("--device", default="1")
    p.add_argument("--timeout-hours", type=float, default=24)
    a = p.parse_args()
    if a.timeout_hours <= 0:
        raise ValueError("Timeout must be positive")
    output = a.new_run / "cross_comparison"
    state_path = a.new_run / "cross_comparison_status.json"

    def state(stage: str, **extra) -> None:
        value = dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(state_path)
        print(json.dumps(value), flush=True)

    deadline = time.monotonic() + a.timeout_hours * 3600
    try:
        state("waiting_for_weights")
        while True:
            status = json.loads((a.new_run / "status.json").read_text())
            stage = status["stage"]
            if stage == "weights_ready":
                break
            if stage == "failed":
                raise RuntimeError(f"Training failed: {status.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Training exceeded comparison wait budget")
            os.kill(int(status["pid"]), 0)
            time.sleep(60)
        state("evaluating")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/anti_uav/compare_cross_expansion.py"),
            "--baseline-run", str(a.baseline_run), "--new-run", str(a.new_run),
            "--data-yaml", str(a.data_yaml), "--output", str(output),
            "--device", a.device,
        ], check=True)
        state("complete", report=str(output / "COMPARISON.md"))
    except Exception as error:
        state("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
