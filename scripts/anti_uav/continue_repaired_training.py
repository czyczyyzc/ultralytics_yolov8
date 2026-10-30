#!/usr/bin/env python3
"""Continue a verified, completed repaired foundation without restarting its epochs."""

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def completed_foundation(training):
    status_path = training/"status.json"
    if not status_path.is_file():
        return None
    state = json.loads(status_path.read_text())
    if state.get("stage") == "failed":
        raise RuntimeError("Foundation failed; refuse to skip an incomplete training stage")
    if state.get("stage") != "foundation_complete":
        return None
    with (training/"foundation/p3/results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(float(next(v for k, v in row.items() if k.strip() == "epoch"))) for row in rows]
    if epochs != list(range(1, 31)):
        raise ValueError("Foundation must complete all 30 epochs before continuation")
    weights = training/"foundation/p3/weights/best.pt"
    metrics = training/"audit/video00009_foundation_p3.json"
    if not weights.is_file() or not metrics.is_file():
        raise ValueError("Completed foundation weights or validation results are missing")
    values = json.loads(metrics.read_text())
    if any(not math.isfinite(values[k]) for k in ("fitness", "native/c0.03/R", "native/mAP50")):
        raise ValueError("Foundation validation contains nonfinite metrics")
    return dict(weights=str(weights), weights_sha256=sha256(weights), metrics=str(metrics),
                metrics_sha256=sha256(metrics), completed_epochs=30)


def delivery_command(python, run, data, delivery, foundation, baseline, toolkit, device):
    return [str(python), "-u", str(ROOT/"scripts/anti_uav/run_labeler_retrain_delivery.py"),
        "--run-dir", str(run), "--data-dir", str(data), "--delivery-dir", str(delivery),
        "--foundation", str(foundation), "--baseline-student", str(baseline),
        "--toolkit-python", str(toolkit), "--device", str(device), "--workers", "16", "--resume"]


def matching_process(pid, script, run):
    path = Path(f'/proc/{pid}/cmdline')
    try:
        command = path.read_bytes().decode().replace('\0', ' ')
    except FileNotFoundError:
        return False
    return script in command and str(run) in command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "data-dir", "delivery-dir", "foundation", "baseline-student", "toolkit-python"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--foundation-parent-pid", type=int, required=True)
    p.add_argument("--foundation-child-pid", type=int, required=True)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    for name in ("run_dir", "data_dir", "delivery_dir", "foundation", "baseline_student", "toolkit_python"):
        setattr(a, name, getattr(a, name).resolve())
    control = a.run_dir/"full_curriculum_continuation"
    control.mkdir(exist_ok=False)
    lock = (control/"continuation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = a.data_dir/"train_online_gray_monitor.yaml"
    manifest = a.data_dir/"manifest.json"
    pinned = {str(path): sha256(path) for path in (config, manifest, a.data_dir/"annotation_snapshot.json", a.foundation)}
    protocol_path = a.run_dir/"training/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("sampling_policy", {}).get("policy") != "balanced_v2" or protocol.get("qat"):
        raise ValueError("Continuation is restricted to the repaired non-QAT curriculum")
    if protocol["foundation_sha256"] != sha256(a.foundation) or protocol["data"] != str(config):
        raise ValueError("Continuation would change the public initialization or frozen data")
    command = delivery_command(sys.executable, a.run_dir, a.data_dir, a.delivery_dir,
        a.foundation, a.baseline_student, a.toolkit_python, a.device)
    dump(control/"protocol.json", dict(pid=os.getpid(), pinned=pinned, command=command,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        preserve_current_foundation=True, restart_foundation=False,
        public_foundation_stage_started_from_scratch=True, teacher_epochs=12, student_epochs=15,
        old_unrepaired_experiment_restarted=False, previous_student_used_for_evaluation_only=True))
    def status(stage, **extra):
        dump(control/"status.json", dict(stage=stage, pid=os.getpid(), **extra))
    try:
        status("waiting_for_repaired_foundation")
        deadline = time.monotonic()+72*3600
        while True:
            state = json.loads((a.run_dir/"workflow_status.json").read_text())
            if state.get("stage") == "failed":
                raise RuntimeError("Repaired foundation workflow failed")
            result = completed_foundation(a.run_dir/"training")
            active = matching_process(a.foundation_parent_pid, "run_balanced_labeler_repair.py", a.run_dir)
            active |= matching_process(a.foundation_child_pid, "run_p3_independent_curriculum_training.py", a.run_dir)
            if result is not None and not active:
                break
            if not active and result is None:
                raise RuntimeError("Foundation processes exited without successful completion")
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for repaired foundation")
            time.sleep(30)
        for path, digest in pinned.items():
            if sha256(path) != digest:
                raise ValueError("Frozen data or public initialization changed before continuation")
        status("waiting_for_gpu_release")
        for _ in range(20):
            usage = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
                "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            memory, utilization = map(int, usage.strip().split(','))
            if memory <= 256 and utilization <= 5:
                break
            time.sleep(30)
        else:
            raise RuntimeError("GPU remained busy; refuse to disturb another task")
        dump(control/"completed_foundation.json", result)
        original = json.loads(protocol_path.read_text())
        dump(control/"foundation_only_training_protocol.json", original)
        dump(control/"foundation_only_workflow_status.json", json.loads((a.run_dir/"workflow_status.json").read_text()))
        dump(protocol_path, dict(original, execution_scope="full_curriculum",
             continuation_audit=str(control/"protocol.json")))
        status("continuing_teacher_student_and_delivery", command=command)
        with (control/"continuation.log").open("a") as log:
            subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4"), check=True)
        final = json.loads((a.run_dir/"workflow_status.json").read_text())
        if final.get("stage") != "complete":
            raise RuntimeError("Full repaired curriculum did not finish successfully")
        status("complete", result=final)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
