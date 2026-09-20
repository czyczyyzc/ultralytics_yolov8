#!/usr/bin/env python3
"""Measure process launch to first result, including interpreter and imports."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.runs < 1:
        p.error("Provide a command and at least one run")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(args.runs):
        trial = args.output / f"trial_{index+1}"
        argv = command + ["--output", str(trial.resolve()), "--frames", "1", "--warmup", "0"]
        first = event = None
        start = time.perf_counter()
        with (args.output / f"trial_{index+1}.log").open("x") as log:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    received = time.perf_counter()
                    log.write(line)
                    try:
                        parsed = json.loads(line)
                    except ValueError:
                        continue
                    if parsed.get("event") == "first_result" and first is None:
                        first, event = (received-start)*1000, parsed
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
        if code or first is None:
            raise RuntimeError(f"Startup trial {index+1} failed: exit={code}")
        summary = json.loads((trial / "summary.json").read_text())
        results.append(dict(trial=index+1, process_launch_to_first_result_ms=first,
                            model_load_ms=summary["model_load_ms"], first_result=event))
        print(json.dumps(results[-1]), flush=True)
    values = [r["process_launch_to_first_result_ms"] for r in results]
    report = dict(trials=results, median_ms=statistics.median(values),
                  min_ms=min(values), max_ms=max(values),
                  scope="Fresh process launch (including shell/interpreter/imports/model load/decode) to first result on stdout. OS and filesystem caches are not reset. Not board power-on or camera/display latency.")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
