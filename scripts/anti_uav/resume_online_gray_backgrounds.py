#!/usr/bin/env python3
"""Scale an existing background-cache job without changing its frozen algorithm plan."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.prepare_online_gray_replacement import (
    atomic_json, build_plan, finalize, prepare_video,
)
from scripts.anti_uav.synthesize_gray_drone_replacements import sha256


def cpu_groups(text, workers, available):
    selected = []
    for entry in text.split(","):
        bounds = entry.split("-")
        if len(bounds) == 1:
            selected.append(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            selected.extend(range(int(bounds[0]), int(bounds[1])+1))
        else:
            raise ValueError("Invalid CPU range")
    if len(set(selected)) != len(selected) or not set(selected) <= set(available):
        raise ValueError("CPU list is duplicated or outside the allowed affinity")
    if not 1 <= workers <= 32 or len(selected) != workers*2:
        raise ValueError("Select exactly two logical CPUs per worker, with 1..32 workers")
    return [selected[i:i+2] for i in range(0, len(selected), 2)]


def unique_physical_cores(groups, topology=Path("/sys/devices/system/cpu")):
    identities = []
    for group in groups:
        for cpu in group:
            root = topology/f"cpu{cpu}"/"topology"
            identities.append(((root/"physical_package_id").read_text().strip(),
                               (root/"core_id").read_text().strip()))
    if len(set(identities)) != len(identities):
        raise ValueError("Select distinct physical cores rather than two SMT siblings")
    return len(identities)


def ensure_quiescent(root, proc=Path("/proc")):
    names = {"prepare_online_gray_replacement.py", "resume_online_gray_backgrounds.py"}
    for path in proc.glob("[0-9]*/cmdline"):
        pid = int(path.parent.name)
        if pid == os.getpid():
            continue
        try:
            args = path.read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(root) in args and any(Path(arg).name in names for arg in args if arg):
            raise RuntimeError(f"Another worker/coordinator is still writing this cache: PID {pid}")


def initialize_worker(groups, counter, root):
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if index >= len(groups):
        raise RuntimeError("Unexpected replacement worker; refusing CPU overlap")
    os.sched_setaffinity(0, groups[index])
    atomic_json(Path(root)/f"worker_{os.getpid()}.json",
                dict(pid=os.getpid(), cpu_ids=groups[index], opencv_threads=2))


def resume(root, workers, cpus):
    groups = cpu_groups(cpus, workers, os.sched_getaffinity(0))
    physical_cores = unique_physical_cores(groups)
    lock = (root/"coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ensure_quiescent(root)
    plan = json.loads((root/"plan.json").read_text())
    live = build_plan(Path(plan["source_dataset"]), Path(plan["catalog"]))
    if live != plan:
        raise ValueError("Frozen annotation/assets/algorithm changed; execution scaling cannot override this check")
    previous = json.loads((root/"job_status.json").read_text())
    if previous.get("is_smoke_subset"):
        raise ValueError("Use a full planned job, not a smoke subset")
    if previous.get("stage") == "complete":
        raise ValueError("This job has already completed")
    before = [json.loads(p.read_text()) for p in root.glob("*.status.json")]
    start = time.time()
    status = dict(stage="running", pid=os.getpid(), total_videos=len(plan["videos"]),
        completed_videos=[], is_smoke_subset=False, training_started=False,
        workers=workers, physical_cores=physical_cores, cpu_groups=groups,
        resumed_at_unix=start, previously_processed=sum(s.get("processed", 0) for s in before),
        previous_pid=previous.get("pid"), dispatcher_sha256=sha256(Path(__file__)),
        frozen_plan_unchanged=True)
    atomic_json(root/f"execution_resume_{int(start)}.json", dict(status, previous_status=previous))
    atomic_json(root/"job_status.json", status)
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    errors = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize_worker,
                                 initargs=(groups, counter, str(root))) as pool:
            futures = {pool.submit(prepare_video, plan, item, str(root)): item["record"]["video"]
                       for item in plan["videos"]}
            for future in as_completed(futures):
                try:
                    status["completed_videos"].append(future.result())
                except Exception as error:
                    errors.append(dict(video=futures[future], error=repr(error)))
                    # Report failed videos immediately while independent workers finish.
                    status.update(stage="running_with_errors", failed_videos=errors)
                atomic_json(root/"job_status.json", status)
        if errors:
            raise RuntimeError(f"{len(errors)} videos failed; see failed_videos")
        summary = finalize(plan, plan["videos"], root, False)
        status.update(stage="complete", summary=summary)
        atomic_json(root/"job_status.json", status)
    except BaseException as error:
        status.update(stage="failed", error=repr(error))
        atomic_json(root/"job_status.json", status)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpus", default="32-63", help="Two distinct physical cores per worker")
    args = parser.parse_args()
    resume(args.output.resolve(), args.workers, args.cpus)


if __name__ == "__main__":
    main()
