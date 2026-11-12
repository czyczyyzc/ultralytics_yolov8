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
    p.add_argument("--stopped-control-snapshot", type=Path, help="Frozen receipt from an interrupted control; compare at its exact step")
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
        stopped = read(a.stopped_control_snapshot) if a.stopped_control_snapshot else None
        if stopped and (stopped["control"] != str(control) or not 1 <= stopped["epoch"] <= epochs
                or stopped["batches"] != batch_count or sha256(stopped["checkpoint"]) != stopped["sha256"]
                or stopped["optimizer_updates"] != scheduled_updates(batch_count, stopped["epoch"], control_args)
                or observed_batches(control, stopped["epoch"]) != [batch_count]*stopped["epoch"]):
            raise ValueError("Stopped control snapshot is not the matched scheduled checkpoint")
        dump(run/"comparison_protocol.json", dict(control=str(control), data=str(data), training_videos=97,
            data_sha256=sha256(data), foundation_data_sha256=sha256(control_config), baseline_protocol_sha256=sha256(control/"protocol.json"),
            batch_quotas=source["tiny_sampling"]["batch_quotas"], foundation_epochs=epochs,
            batches_per_epoch=batch_count, optimizer_updates_scheduled=updates,
            candidate="P3 signed DFL + representable fallback + matched-only regression budget",
            checkpoints="last at equal step budget", stopped_control=stopped,
            validation="Video00009, FP32, native frames, no zoom",
            eval_budgets_fp_per_1000=[100,200,400], teacher_or_student_automatically_started=False))
        gpu = wait_idle()
        command = training_command(baseline, data, foundation, candidate, gpu, 97)
        command += ["--foundation-only","--p3-signed-dfl"]
        status("train_p3_initial", gpu=gpu, training_videos=97, batches_per_epoch=batch_count,
               matched_epoch=stopped["epoch"] if stopped else epochs)
        comparison = None
        if stopped:
            command += ["--foundation-compare-epoch", str(stopped["epoch"])]
            receipt_path = candidate/"audit/matched_epoch/snapshot.json"
            comparison = run/f"comparison_epoch_{stopped['epoch']:04d}"
            with (logs/"train_p3_initial.log").open("x") as stream:
                trainer = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            evaluation_error = None
            while not receipt_path.is_file() and trainer.poll() is None:
                time.sleep(10)
            if receipt_path.is_file():
                try:
                    saved = read(receipt_path)
                    if (any(saved[key] != stopped[key] for key in ("epoch","optimizer_updates","batches"))
                            or sha256(saved["checkpoint"]) != saved["sha256"]):
                        raise ValueError("Comparison checkpoints are not at the same optimizer step")
                    status("evaluate_matched_epoch_while_training", gpu=gpu, matched_epoch=stopped["epoch"])
                    run_command("evaluate_matched_epoch", [sys.executable,"-u","-m","scripts.anti_uav.evaluate_p3_equal_fp",
                        "--control",stopped["checkpoint"],"--offset",saved["checkpoint"],"--candidate-label","p3_signed_dfl",
                        "--device","cpu","--video00009",data,"--output",comparison])
                    status("matched_epoch_evaluated_training_continues", gpu=gpu, matched_epoch=stopped["epoch"],
                           comparison=str(comparison/"comparison.json"))
                except Exception as error:
                    evaluation_error = error
                    status("matched_evaluation_failed_training_continues", gpu=gpu, error=repr(error))
            if trainer.wait() != 0:
                raise subprocess.CalledProcessError(trainer.returncode, command)
            if evaluation_error:
                raise evaluation_error
            if not (comparison/"comparison.json").is_file():
                raise ValueError("Candidate finished without the matched checkpoint evaluation")
        else:
            run_command("train_p3_initial", command)
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
        if comparison:
            status("complete_awaiting_teacher_student_decision", matched_epoch=stopped["epoch"],
                   comparison=str(comparison/"comparison.json"), epoch_30_control_unavailable=stopped["epoch"]<epochs)
            return
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
