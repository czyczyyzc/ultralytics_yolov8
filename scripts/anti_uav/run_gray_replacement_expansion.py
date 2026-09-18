#!/usr/bin/env python3
"""Run expanded synthesis then a full integrity audit; never start model training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New job directory; batch is inside candidates/")
    p.add_argument("--exclude-batch", type=Path, action="append", default=[])
    p.add_argument("--per-video", type=int, default=100)
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    def status(stage, **extra):
        path = a.output / "job_status.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(stage=stage, pid=os.getpid(),
            updated_at=datetime.now(timezone.utc).isoformat(), training_started=False, **extra), indent=2)+"\n")
        tmp.replace(path)
    command = [sys.executable, str(scripts / "build_gray_replacement_batch.py"),
        "--dataset", str(a.dataset), "--catalog", str(a.catalog), "--include-legacy",
        "--per-video", str(a.per_video), "--variants", str(a.variants), "--previews", "48", "--seed", str(a.seed)]
    for batch in a.exclude_batch:
        command += ["--exclude-batch", str(batch)]
    try:
        status("planning")
        subprocess.run(command + ["--plan-only", "--output", str(a.output / "plan")], check=True)
        plan = json.loads((a.output / "plan/plan.json").read_text())
        if plan["missing_training_hashes"]:
            raise ValueError("Not all declared training videos have source adapters")
        status("generating", selected_frames=plan["selected_frames"], maximum_outputs=plan["maximum_outputs"])
        subprocess.run(command + ["--output", str(a.output / "candidates")], check=True)
        status("verifying")
        subprocess.run([sys.executable, str(scripts / "verify_gray_replacement_batch.py"),
                        "--batch", str(a.output / "candidates"), "--report", str(a.output / "verification.json")], check=True)
        summary = json.loads((a.output / "candidates/summary.json").read_text())
        status("complete", accepted=summary["accepted"], source_videos=summary["source_videos_replaced"],
               manual_visual_review_pending=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
