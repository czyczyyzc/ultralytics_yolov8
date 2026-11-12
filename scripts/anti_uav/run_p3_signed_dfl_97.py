"""Train signed P3 only against the unmodified 97-video box-preserving foundation."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
from scripts.anti_uav.run_approved_gray_fixedquota_refresh import choose_shared_gpu, training_command
from scripts.anti_uav.run_p3_independent_curriculum_training import audit_label_caches, stage_yaml

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def observed_batches(arm, epochs):
    path = arm/"foundation/p3/regression_repair_audit"
    return [read(path/f"epoch_{epoch:04d}.json")["losses"]["detector"]["batches"] for epoch in range(1, epochs+1)]


def scheduled_updates(batches, epochs, args):
    warmup = max(round(args["warmup_epochs"]*batches), 100) if args["warmup_epochs"] > 0 else -1
    ratio = args["nbs"]/args["batch"]
    count, last = 0, -1
    accumulate = max(round(ratio), 1)
    for i in range(batches*epochs):
        if i <= warmup:
            accumulate = max(round(1+(ratio-1)*i/warmup), 1)
        if i-last >= accumulate:
            count += 1
            last = i
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    a = p.parse_args()
    control, run = a.control.resolve(), a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run/"signed_dfl.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = run/"logs"
    logs.mkdir(exist_ok=True)
    candidate = run/"candidate"

    def status(stage, **more):
        dump(run/"status.json", dict(stage=stage, pid=os.getpid(), updated=time.time(), **more))

    def run_command(name, command):
        with (logs/f"{name}.log").open("x") as stream:
            subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def wait_idle():
        while True:
            usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True)
            gpu = choose_shared_gpu(usage, a.gpus, False)
            if gpu is not None:
                return gpu
            status("waiting_for_idle_gpu", usage=usage, no_sharing=True)
            time.sleep(60)

    try:
        status("checking_frozen_control")
        baseline = read(control/"protocol.json")
        preparation = read(control/"preparation.json")
        data, foundation = Path(baseline["data"]), Path(baseline["foundation"])
        source = yaml.safe_load(data.read_text())
        epochs = baseline["epochs"]["foundation"]
        if (baseline["training_videos"] != 97 or preparation["training_videos"] != 97 or epochs != 30
                or source["visible_box_guard"] != "affine_keep_subpixel_v2" or not preparation["complete"]
                or len(read(data.parent/"manifest.json")["train_video_hashes"]) != 97):
            raise ValueError("Not the requested 97-video box-preserving control")
        if (sha256(data) != preparation["data_sha256"] or sha256(foundation) != baseline["foundation_sha256"]
                or sha256(Path(source["online_replacement"]["cache"])/"index.json") != preparation["backgrounds"]["index_sha256"]):
            raise ValueError("Frozen control data, assets, or public initialization changed")
        if audit_label_caches(source) != baseline["trusted_label_caches"]:
            raise ValueError("Training or validation list/cache differs from the control")
        config = stage_yaml(source, run/"foundation_contract.yaml", 0, baseline["augmentation"]["online_replacement_probability"])
        control_config = control/"configs/foundation.yaml"
        if yaml.safe_load(config.read_text()) != yaml.safe_load(control_config.read_text()):
            raise ValueError("Candidate would not use the exact same effective foundation data and sampler")
        control_args = yaml.safe_load((control/"foundation/p3/args.yaml").read_text())
        launch = read(control/"launch.json")
        if launch.get("resuming") or launch["command"] != training_command(baseline, data, foundation, control, launch["gpu"], 97):
            raise ValueError("Control is not the matched audited launch")
        batch_count = observed_batches(control, 1)[0]
        updates = scheduled_updates(batch_count, epochs, control_args)
        dump(run/"comparison_protocol.json", dict(control=str(control), data=str(data), training_videos=97,
            data_sha256=sha256(data), foundation_data_sha256=sha256(control_config), baseline_protocol_sha256=sha256(control/"protocol.json"),
            batch_quotas=source["tiny_sampling"]["batch_quotas"], foundation_epochs=epochs,
            batches_per_epoch=batch_count, optimizer_updates_scheduled=updates,
            candidate="P3 signed DFL + representable fallback + matched-only regression budget",
            checkpoints="last at equal step budget", validation="Video00009, FP32, native frames, no zoom",
            eval_budgets_fp_per_1000=[100,200,400], teacher_or_student_automatically_started=False))
        gpu = wait_idle()
        command = training_command(baseline, data, foundation, candidate, gpu, 97)
        status("train_p3_initial", gpu=gpu, training_videos=97, batches_per_epoch=batch_count)
        run_command("train_p3_initial", command+["--foundation-only","--p3-signed-dfl"])
        trained = read(candidate/"status.json")
        if (trained["stage"] != "foundation_complete" or trained["epoch"] != epochs
                or trained["batches_per_epoch"] != batch_count or trained["optimizer_updates"] != updates):
            raise ValueError("Candidate did not complete the control's scheduled step budget")
        trained_args = yaml.safe_load((candidate/"foundation/p3/args.yaml").read_text())
        run_specific = {"device","data","project","name","save_dir"}
        for key in (set(control_args) | set(trained_args))-run_specific:
            if control_args.get(key) != trained_args.get(key):
                raise ValueError(f"Control and candidate differ in training argument {key}")
        if yaml.safe_load((candidate/"configs/foundation.yaml").read_text()) != yaml.safe_load(control_config.read_text()):
            raise ValueError("Candidate training config changed")
        while True:
            control_status = read(control/"status.json")
            if control_status["stage"] not in {"train_native_p3_foundation","initialize_foundation"}:
                break
            status("waiting_for_control_foundation", control_epoch=control_status.get("epoch"))
            time.sleep(60)
        for arm in (control, candidate):
            with (arm/"foundation/p3/results.csv").open() as stream:
                recorded = [int(float(row["epoch"])) for row in csv.DictReader(stream, skipinitialspace=True)]
            if recorded != list(range(1, epochs+1)) or observed_batches(arm, epochs) != [batch_count]*epochs:
                raise ValueError(f"Missing or mismatched foundation history: {arm}")
        weights = "foundation/p3/weights/last.pt"
        gpu = wait_idle()
        status("evaluate_video00009_equal_fp", gpu=gpu)
        run_command("evaluate_video00009_equal_fp", [sys.executable,"-u","-m","scripts.anti_uav.evaluate_p3_equal_fp",
            "--control",control/weights,"--offset",candidate/weights,"--candidate-label","p3_signed_dfl",
            "--device",gpu,"--video00009",data,"--output",run/"comparison"])
        status("complete_awaiting_teacher_student_decision", comparison=str(run/"comparison/comparison.json"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
