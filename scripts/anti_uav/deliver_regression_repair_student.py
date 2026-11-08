#!/usr/bin/env python3
"""Wait for the repaired curriculum, then release its verified RK3576 P3 binary."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
from scripts.anti_uav.run_labeler_retrain_delivery import completed_student_metadata


def read(path):
    return json.loads(Path(path).read_text())


def validate_inputs(training):
    protocol = read(training / "protocol.json")
    contract = read(training / "experiment_contract.json")
    if (not protocol.get("regression_repair") or protocol.get("qat")
            or protocol.get("input_color") != "gray3" or protocol.get("training_videos") != 74
            or protocol.get("input_hw") != [544, 960] or protocol.get("task_specific_initialization")):
        raise ValueError("Require the repaired public-initialized 74-video gray3 FP32 curriculum")
    if contract["same_data"] != protocol["data"] or contract["training_videos"] != 74:
        raise ValueError("Repair data contract differs from actual training")
    for path, expected in contract["pinned"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Pinned training input changed: {path}")
    for listing, pin in protocol["trusted_label_caches"].items():
        if sha256(Path(listing)) != pin["source_sha256"] or sha256(Path(pin["path"])) != pin["cache_sha256"]:
            raise ValueError(f"Pinned train/validation list or label cache changed: {listing}")
    data = yaml.safe_load(Path(protocol["data"]).read_text())
    if data.get("input_color") != "gray3" or data["tiny_sampling"]["policy"] != "fixed_domain_v1":
        raise ValueError("Expected fixed-quota gray3 data")
    return protocol, contract, data


def ready_state(training):
    state = read(training / "status.json")
    supervisor = read(training / "repair_launch_status.json")
    if state["stage"] == "failed" or supervisor["stage"] == "failed":
        raise RuntimeError(f"Training/comparison failed: {state}; {supervisor}")
    return state["stage"] == supervisor["stage"] == "complete", state, supervisor


def evaluated_binary(evaluation):
    if read(evaluation / "status.json")["stage"] != "complete":
        raise ValueError("Quantization evaluation is incomplete")
    result = read(evaluation / "results.json")
    protocol = result["protocol"]
    binary = evaluation / "rebuilt_for_simulator.rknn"
    if (protocol["target"] != "rk3576" or protocol.get("input_color") != "gray3"
            or protocol["quantized_dtype"] != "w8a8" or protocol.get("auto_hybrid")
            or sha256(binary) != protocol["rebuilt_sha256"]):
        raise ValueError("Evaluated binary is not the requested gray3 RK3576 W8A8 artifact")
    expected = {"Video00004_test": 100, "Video00009_selection_subset": 100}
    for split, count in expected.items():
        if protocol["datasets"][split]["evaluated_frames"] != count:
            raise ValueError("Quantization smoke coverage changed")
        for backend in ("pt_fp32", "onnx_fp32", "rknn_int8_simulator"):
            if result["backends"][backend][split]["native/c0.03/FRAMES"] != count:
                raise ValueError("Backend quantization coverage differs")
    if result["raw_export_max_error"]["class_score"] > 1e-4:
        raise ValueError("ONNX confidence parity failed; do not ship an unverified export")
    return binary, result


def metric_table(records, split, names):
    lines = ["| 模型 | Recall | Precision | FP | mAP50 | mAP50-95 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in names.items():
        record = records[name]
        metrics = record["splits"][split] if "splits" in record else record[split]
        q = "native/c0.03/"
        lines.append(f"| {label} | {metrics[q+'R']:.2%} | {metrics[q+'P']:.2%} | {metrics[q+'FP']} | "
                     f"{metrics['native/mAP50']:.2%} | {metrics['native/mAP50-95']:.2%} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--delivery-dir", type=Path, required=True)
    parser.add_argument("--toolkit-python", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-hours", type=float, default=48)
    args = parser.parse_args()
    for key in ("training", "work_dir", "delivery_dir", "toolkit_python"):
        setattr(args, key, getattr(args, key).resolve())
    if args.poll_seconds < 30 or args.wait_hours <= 0:
        parser.error("Require a bounded, nonaggressive wait")
    if args.work_dir == args.training or args.delivery_dir.exists():
        raise ValueError("Do not overwrite training or an existing delivery")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    lock = (args.work_dir / "delivery.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = args.work_dir / "logs"
    logs.mkdir()
    environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
                       MPLBACKEND="Agg", WANDB_MODE="disabled", PYTHONPATH=str(ROOT))
    environment.pop("CUDA_VISIBLE_DEVICES", None)

    def status(stage, **extra):
        record = dict(stage=stage, pid=os.getpid(), time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra)
        dump(args.work_dir / "workflow_status.json", record)
        print(json.dumps(record), flush=True)

    def execute(stage, command):
        status(stage, command=command)
        with (logs / (stage + ".log")).open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        status("preflight")
        protocol, contract, data = validate_inputs(args.training)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        version = subprocess.check_output([str(args.toolkit_python), "-c",
            "from importlib.metadata import version; print(version('rknn-toolkit2'))"], text=True).strip()
        if version != "2.3.2":
            raise ValueError(f"Expected RKNN Toolkit2 2.3.2, got {version}")
        plan = dict(training=str(args.training), data=protocol["data"], target="rk3576", toolkit_version=version,
                    training_videos=74, input_hw=[544, 960], input_color="gray3", qat=False,
                    quantization="W8A8 PTQ, normal/channel", git_commit=commit,
                    protocol_sha256=sha256(args.training / "protocol.json"),
                    contract_sha256=sha256(args.training / "experiment_contract.json"),
                    code_snapshot="committed release only; unrelated working-tree edits excluded",
                    board_validation_complete=False, commercial_dataset_authorizations_verified=False)
        dump(args.work_dir / "workflow_protocol.json", plan)
        calibration = args.work_dir / "calibration"
        execute("prepare_calibration", [sys.executable, str(ROOT / "scripts/anti_uav/prepare_stratified_rknn_calibration.py"),
            "--data", protocol["data"], "--output-dir", str(calibration), "--samples", "384",
            "--exclude-token", "Video00004", "Video00009", "zoom_val", "zoom_train"])
        audit = read(calibration / "audit.json")
        if audit.get("input_color") != "gray3" or audit["samples"] != 384:
            raise ValueError("Calibration does not match gray3 deployment")
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            ready, state, supervisor = ready_state(args.training)
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Training/comparison did not complete within the delivery wait budget")
            status("waiting_for_completed_student", training_stage=state["stage"], epoch=state.get("epoch"),
                   comparison_stage=supervisor["stage"], training_pid=state.get("pid"))
            time.sleep(args.poll_seconds)
        validate_inputs(args.training)
        if (sha256(args.training / "protocol.json") != plan["protocol_sha256"]
                or sha256(args.training / "experiment_contract.json") != plan["contract_sha256"]):
            raise ValueError("Training identity changed while waiting")
        completed = completed_student_metadata(args.training)
        student = Path(completed["student"])
        comparison = read(args.training / "comparison/results.json")
        selected = comparison["approved74_regression_repair_student"]
        if selected["sha256"] != completed["student_sha256"] or Path(selected["weights"]).resolve() != student:
            raise ValueError("Final FP32 comparison does not cover the selected pure-P3 Student")
        onnx = args.work_dir / "detector_960x544_rkopt.onnx"
        execute("export_onnx", [sys.executable, str(ROOT / "scripts/anti_uav/export_detector_rkopt_onnx.py"),
            "--weights", str(student), "--imgsz", "544,960", "--device", "cpu", "--output", str(onnx)])
        shapes = read(onnx.with_suffix(".rkopt.json"))["validation"]
        if shapes["output_count"] != 9 or shapes["branch_count"] != 3:
            raise ValueError("Deployment graph still contains auxiliary P2")
        reference = args.work_dir / "reference_rk3576_w8a8.rknn"
        execute("build_rk3576_int8", [str(args.toolkit_python), str(ROOT / "scripts/anti_uav/build_rknn.py"),
            "--onnx", str(onnx), "--output", str(reference), "--target", "rk3576", "--quantize",
            "--dataset", str(calibration / "dataset.txt"), "--quantized-dtype", "w8a8",
            "--quantized-algorithm", "normal", "--quantized-method", "channel"])
        evaluation = args.work_dir / "quantization_smoke"
        execute("quantization_smoke", [str(args.toolkit_python), "-u",
            str(ROOT / "scripts/anti_uav/compare_gray_rknn_quantization.py"), "--weights", str(student),
            "--onnx", str(onnx), "--rknn", str(reference), "--calibration", str(calibration / "dataset.txt"),
            "--target", "rk3576", "--input-color", "gray3", "--device", "cpu",
            "--output", str(evaluation), "--limit-per-split", "100"])
        tested, quantized = evaluated_binary(evaluation)
        execute("load_rk3576", [str(args.toolkit_python), "-c",
            "from rknn.api import RKNN; import sys; r=RKNN(); v=r.load_rknn(sys.argv[1]); "
            "print('load_rknn_ret=',v); r.release(); assert v==0", str(tested)])
        if sha256(student) != completed["student_sha256"]:
            raise ValueError("Selected checkpoint changed during export")
        status("packaging")
        package = args.delivery_dir
        for name in ("models", "metadata", "source"):
            (package / name).mkdir(parents=True, exist_ok=False)
        pairs = [(student, "models/p3_student_best_fp32.pt"), (onnx, "models/detector_960x544_rkopt.onnx"),
                 (tested, "models/detector_960x544_rk3576_w8a8.rknn"),
                 (onnx.with_suffix(".rkopt.json"), "metadata/detector.rkopt.json"),
                 (calibration / "audit.json", "metadata/calibration_audit.json"),
                 (calibration / "dataset.txt", "metadata/calibration_dataset.txt"),
                 (args.work_dir / "workflow_protocol.json", "metadata/workflow_protocol.json"),
                 (Path(protocol["data"]), "metadata/train_fixed_gray.yaml")]
        for relative in ("protocol.json", "selection.json", "experiment_contract.json", "audit/student_exports.json",
                         "comparison/results.json", "comparison/protocol.json"):
            pairs.append((args.training / relative, "metadata/training/" + relative))
        for source, relative in pairs:
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copytree(args.training / "configs", package / "metadata/training/configs")
        for stage in ("foundation/p3", "teacher/p2p3", "distill_fp32/p3_student"):
            target = package / "metadata/training" / stage
            target.mkdir(parents=True)
            shutil.copy2(args.training / stage / "results.csv", target / "results.csv")
        shutil.copytree(evaluation, package / "metadata/quantization_smoke",
                        ignore=shutil.ignore_patterns("*.rknn", "metrics"))
        dump(package / "metadata/quantization_smoke/delivered_binary.json", dict(
            path="models/detector_960x544_rk3576_w8a8.rknn", sha256=sha256(tested),
            evaluated_sha256=quantized["protocol"]["rebuilt_sha256"], final_binary_is_simulator_evaluated=True))
        subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(package / "source/code.tar.gz"),
            commit, "LICENSE", "scripts/anti_uav/dist_native", "scripts/anti_uav/rknn_yolov8_native",
            "scripts/anti_uav/build_dist_detector_on_board.sh"], cwd=ROOT, check=True)
        artifact = dict(plan, created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            delivery_dir=str(package), source_pt=str(student), source_best_epoch=completed["best_epoch"],
            source_pt_sha256=sha256(student), onnx_sha256=sha256(onnx), model_sha256=sha256(tested),
            quantization_evaluated_frames=200, final_binary_is_simulator_evaluated=True,
            production_accepted=False)
        dump(package / "FINAL_ARTIFACT.json", artifact)
        names = dict(native69_student="旧69视频 P3 Student", approved74_legacy_loss_student="74视频未修复 P3 Student",
                     approved74_regression_repair_student="本次74视频修复版 P3 Student")
        values = dict(CREATED_AT=artifact["created_at"], DELIVERY_DIR=str(package), TRAINING_DIR=str(args.training),
            SOURCE_PT=str(student), DATA=str(protocol["data"]), BEST_EPOCH=str(completed["best_epoch"]), COMMIT=commit,
            MODEL_SHA256=artifact["model_sha256"], PT_SHA256=artifact["source_pt_sha256"], ONNX_SHA256=artifact["onnx_sha256"],
            FULL_METRICS=metric_table(comparison, "pooled", names),
            QUANT_METRICS=metric_table(quantized["backends"], "pooled_native",
                dict(pt_fp32="FP32", onnx_fp32="ONNX FP32", rknn_int8_simulator="RKNN INT8 宿主机验证")),
            AUGMENTATION=json.dumps(protocol["augmentation"], ensure_ascii=False, indent=2),
            QUOTAS=json.dumps(protocol["sampling_policy"]["batch_quotas"], ensure_ascii=False),
            REPAIR=json.dumps(protocol["regression_repair"], ensure_ascii=False, indent=2))
        document = (ROOT / "scripts/anti_uav/templates/RK3576_REPAIR_HANDOVER_ZH.md").read_text()
        for key, value in values.items():
            document = document.replace("{{" + key + "}}", value)
        if "{{" in document:
            raise ValueError("Unresolved delivery document field")
        for name in ("DEPLOYMENT_HANDOVER_ZH.md", "README.md"):
            (package / name).write_text(document)
        files = sorted(path for path in package.rglob("*") if path.is_file())
        (package / "SHA256SUMS").write_text("".join(sha256(path) + "  " + str(path.relative_to(package)) + "\n" for path in files))
        archive = package.with_name(package.name + ".tar.gz")
        if archive.exists():
            raise FileExistsError(archive)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(package, arcname=package.name)
        archive.with_name(archive.name + ".sha256").write_text(sha256(archive) + "  " + archive.name + "\n")
        status("complete", archive=str(archive), sha256=sha256(archive),
               handover=str(package / "DEPLOYMENT_HANDOVER_ZH.md"),
               model=str(package / "models/detector_960x544_rk3576_w8a8.rknn"))
    except BaseException as error:
        previous = read(args.work_dir / "workflow_status.json")
        status("failed", failed_stage=previous["stage"], error=repr(error))
        raise


if __name__ == "__main__":
    main()
