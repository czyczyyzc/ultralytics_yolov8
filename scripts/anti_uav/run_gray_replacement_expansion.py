#!/usr/bin/env python3
"""Run expanded synthesis then a full integrity audit; never start model training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument("--output", type=Path, required=True, help="New job directory containing isolated shard_NN/ batches")
    p.add_argument("--exclude-batch", type=Path, action="append", default=[])
    p.add_argument("--per-video", type=int, default=100)
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--workers", type=int, default=4, help="Independent video shards; each OpenCV worker uses two CPU threads")
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        raise ValueError("Worker count must be between 1 and 8")
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
        workers = min(a.workers, len(plan["videos"]))
        status("generating", selected_frames=plan["selected_frames"], maximum_outputs=plan["maximum_outputs"], workers=workers)
        def run_shard(index):
            hashes = [row["sha256"] for row in plan["videos"][index::workers]]
            output = a.output / f"shard_{index:02d}"
            with (a.output / f"shard_{index:02d}.log").open("x") as log:
                subprocess.run(command + ["--output", str(output), "--only-video-sha256", ",".join(hashes)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([sys.executable, str(scripts / "verify_gray_replacement_batch.py"),
                    "--batch", str(output), "--report", str(output / "verification.json")],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            return output
        outputs = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(run_shard, i) for i in range(workers)]):
                outputs.append(future.result())
                status("generating_and_verifying", workers=workers, completed_shards=len(outputs))
        seen, records, source_frames, videos = set(), [], set(), set()
        for output in sorted(outputs):
            data = json.loads((output / "manifest.json").read_text())
            for row in data["samples"]:
                if row["output_sha256"] in seen:
                    continue
                seen.add(row["output_sha256"])
                records.append(str(output / row["image"]))
                source_frames.add((row["video_sha256"], row["frame"]))
                videos.add(row["video_sha256"])
        (a.output / "train_synthetic_unique.txt").write_text("".join(p+"\n" for p in records))
        summary = dict(unique_images=len(records), source_frames=len(source_frames), source_videos=len(videos),
            registered_training_videos=plan["registered_training_videos"], shards=list(map(str, sorted(outputs))),
            selected_frames=plan["selected_frames"], maximum_outputs=plan["maximum_outputs"],
            manual_visual_review_pending=True, training_started=False)
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        status("complete", **{k:v for k,v in summary.items() if k != "training_started"})
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
