#!/usr/bin/env python3
"""Run public-checkpoint retraining, paired evaluation and a verified RK3576 delivery."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def training_command(python, data, foundation, run, device, videos, workers=16):
    return [str(python), "-u", str(ROOT / "scripts/anti_uav/run_p3_independent_curriculum_training.py"),
        "--data", str(data), "--foundation", str(foundation), "--run-dir", str(run),
        "--device", str(device), "--batch", "64", "--workers", str(workers),
        "--foundation-epochs", "30", "--teacher-epochs", "12", "--student-epochs", "15",
        "--localization-epochs", "0", "--replacement-probability", "0.5",
        "--expected-training-videos", str(videos)]


def metric_table(records):
    lines = ["| Model | Recall | Precision | FP | FN | mAP50 | mAP50-95 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in records.items():
        m, q = record["splits"]["pooled_native"], "native/c0.03/"
        lines.append(f"| {name} | {m[q+'R']:.2%} | {m[q+'P']:.2%} | {m[q+'FP']} | {m[q+'FN']} | "
                     f"{m['native/mAP50']:.2%} | {m['native/mAP50-95']:.2%} |")
    return "\n".join(lines)


def augmentation_policy(manifest, state):
    text = "沿用328张候选贴图，在线替换概率0.5，仅对具备安全背景缓存的帧执行。"
    if state.get("cache_ready"):
        ready = state["new_ready_backgrounds"]
        retained = state["added_positive"]-ready
        text += f"本次新增正样本均已检查：{ready}帧有安全背景缓存，另{retained}帧保留原图和原框，不强制替换。"
    else:
        text += "没有背景缓存的新视频使用原图，不宣称新增视频已做全量换贴图。"
    sampler = manifest.get("sampler_audit", {})
    if sampler.get("policy") == "balanced_v2":
        text += (f"采样使用balanced_v2，按增强前960×544长边分类，4–8 px占比约{sampler['tiny_fraction']:.1%}，"
                 f"负样本约{sampler['negative_fraction']:.1%}；保留原正样本曝光，每张图额外重采样最多"
                 f"{sampler['max_extra_repeat_per_image']}次。来源按全轮分散排列，不再排空短视频后集中读取长视频。")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--delivery-dir", type=Path, required=True)
    p.add_argument("--foundation", type=Path, required=True)
    p.add_argument("--baseline-student", type=Path, required=True)
    p.add_argument("--toolkit-python", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--preparation-pid", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    for name in ("data_dir", "run_dir", "delivery_dir", "foundation", "baseline_student", "toolkit_python"):
        setattr(a, name, getattr(a, name).resolve())
    if not a.device.isdecimal():
        raise ValueError("The audited sampler/distiller requires one physical GPU ID")
    if a.delivery_dir.exists():
        raise FileExistsError("Never overwrite a delivery package")
    a.run_dir.mkdir(parents=True, exist_ok=a.resume)
    (a.run_dir / "logs").mkdir(exist_ok=a.resume)
    lock = (a.run_dir / "delivery_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
        MPLBACKEND="Agg", WANDB_MODE="disabled", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT))
    environment.pop("CUDA_VISIBLE_DEVICES", None)

    def status(stage, **extra):
        dump(a.run_dir / "workflow_status.json", dict(stage=stage, pid=os.getpid(),
             time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra))
        print(json.dumps(dict(stage=stage, **extra)), flush=True)

    def execute(stage, command, marker=None):
        if a.resume and marker is not None and marker.is_file():
            return
        status(stage, command=command)
        with (a.run_dir / "logs" / (stage + ".log")).open("a") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        status("waiting_for_prepared_dataset")
        deadline = time.monotonic() + 7200
        while True:
            state_path = a.data_dir / "status.json"
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            if state.get("stage") == "complete":
                break
            if state.get("stage") == "failed" or time.monotonic() > deadline:
                raise RuntimeError(f"Dataset preparation did not complete: {state}")
            if a.preparation_pid:
                try:
                    os.kill(a.preparation_pid, 0)
                except ProcessLookupError:
                    raise RuntimeError("Dataset preparation process exited before completion")
            time.sleep(20)
        manifest = json.loads((a.data_dir / "manifest.json").read_text())
        videos = len(manifest["train_video_hashes"])
        data = a.data_dir / "train_online_gray_monitor.yaml"
        if set(manifest["train_video_hashes"]) & {manifest["test_sha256"], manifest["validation_sha256"]}:
            raise ValueError("Holdout leakage in expanded dataset")
        usage = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, usage.strip().split(","))
        if memory > 256 or utilization > 5:
            raise RuntimeError(f"GPU {a.device} is no longer idle: {usage}")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        plan = dict(training_videos=videos, data=str(data), dataset_manifest_sha256=sha256(a.data_dir / "manifest.json"),
            source_foundation=str(a.foundation), foundation_sha256=sha256(a.foundation),
            baseline_student=str(a.baseline_student), baseline_sha256=sha256(a.baseline_student),
            initialization="Public YOLOv8n only; previous Student used only for evaluation",
            epochs=dict(foundation=30, teacher=12, student=15), qat=False, input_hw=[544, 960],
            batch=64, nbs=128, device=a.device, git_commit=commit,
            source_statuses=state["source_statuses"], added_positive=state["added_positive"], added_negative=state["added_negative"],
            deployment_target="rk3576", quantization="W8A8 PTQ INT8", board_validation_complete=False)
        dump(a.run_dir / "workflow_protocol.json", plan)
        train_dir = a.run_dir / "training"
        command = training_command(sys.executable, data, a.foundation, train_dir, a.device, videos, a.workers)
        if a.resume:
            command.append("--resume")
        execute("10_training", command, train_dir / "selection.json")
        student = train_dir / "p3_student_distilled_best_fp32.pt"
        teacher = train_dir / "teacher/p2p3/weights/best.pt"
        if json.loads((train_dir / "status.json").read_text())["stage"] != "complete":
            raise RuntimeError("Training did not finish successfully")
        models = dict(old52_student=str(a.baseline_student), new_teacher=str(teacher), new_student=str(student))
        dump(a.run_dir / "comparison_models.json", models)
        execute("20_fp32_comparison", [sys.executable, "-u", str(ROOT / "scripts/anti_uav/evaluate_expanded_checkpoint_pair.py"),
            "--models", str(a.run_dir / "comparison_models.json"), "--validation-data", str(data),
            "--output", str(a.run_dir / "evaluation"), "--device", a.device], a.run_dir / "evaluation/results.json")
        export = a.run_dir / "export"
        export.mkdir(exist_ok=True)
        onnx = export / "detector_960x544_rkopt.onnx"
        execute("30_export_onnx", [sys.executable, str(ROOT / "scripts/anti_uav/export_detector_rkopt_onnx.py"),
            "--weights", str(student), "--imgsz", "544,960", "--device", "cpu", "--output", str(onnx)],
            onnx.with_suffix(".rkopt.json"))
        shapes = json.loads(onnx.with_suffix(".rkopt.json").read_text())
        if shapes["validation"]["output_count"] != 9 or shapes["validation"]["branch_count"] != 3:
            raise ValueError("Student export is not a pure three-scale P3 graph")
        calibration = export / "calibration"
        execute("31_calibration", [sys.executable, str(ROOT / "scripts/anti_uav/prepare_stratified_rknn_calibration.py"),
            "--data", str(data), "--output-dir", str(calibration), "--samples", "384",
            "--exclude-token", "Video00004", "Video00009", "zoom_val", "zoom_train"], calibration / "audit.json")
        reference = export / "reference_rk3576_w8a8.rknn"
        execute("32_rk3576_int8_build", [str(a.toolkit_python), str(ROOT / "scripts/anti_uav/build_rknn.py"),
            "--onnx", str(onnx), "--output", str(reference), "--target", "rk3576", "--quantize",
            "--dataset", str(calibration / "dataset.txt"), "--quantized-dtype", "w8a8",
            "--quantized-algorithm", "normal", "--quantized-method", "channel"], reference)
        smoke = export / "quantization_smoke"
        execute("33_quantization_smoke", [str(a.toolkit_python), "-u", str(ROOT / "scripts/anti_uav/compare_gray_rknn_quantization.py"),
            "--weights", str(student), "--onnx", str(onnx), "--rknn", str(reference),
            "--calibration", str(calibration / "dataset.txt"), "--target", "rk3576",
            "--output", str(smoke), "--device", "cuda:" + a.device, "--limit-per-split", "100"], smoke / "REPORT.md")
        quantized = json.loads((smoke / "results.json").read_text())
        tested = smoke / "rebuilt_for_simulator.rknn"
        if (json.loads((smoke / "status.json").read_text())["stage"] != "complete" or
                sha256(tested) != quantized["protocol"]["rebuilt_sha256"] or quantized["protocol"]["target"] != "rk3576"):
            raise ValueError("Actual simulator-tested RK3576 model could not be verified")
        execute("34_load_check", [str(a.toolkit_python), "-c",
            "from rknn.api import RKNN; import sys; r=RKNN(); v=r.load_rknn(sys.argv[1]); print('load_rknn_ret=',v); r.release(); assert v==0",
            str(tested)])
        status("40_packaging")
        package = a.delivery_dir
        package.mkdir(parents=True)
        for name in ("models", "metadata", "source"):
            (package / name).mkdir()
        for source, destination in ((student, package / "models/p3_student_best_fp32.pt"),
                (onnx, package / "models/detector_960x544_rkopt.onnx"),
                (tested, package / "models/detector_960x544_rk3576_w8a8.rknn"),
                (onnx.with_suffix(".rkopt.json"), package / "metadata/detector.rkopt.json"),
                (a.run_dir / "workflow_protocol.json", package / "metadata/workflow_protocol.json"),
                (train_dir / "protocol.json", package / "metadata/training_protocol.json"),
                (train_dir / "selection.json", package / "metadata/selection.json"),
                (a.data_dir / "manifest.json", package / "metadata/data_manifest.json"),
                (a.data_dir / "status.json", package / "metadata/data_preparation.json"),
                (a.run_dir / "evaluation/results.json", package / "metadata/fp32_comparison.json"),
                (calibration / "audit.json", package / "metadata/calibration_audit.json"),
                (calibration / "dataset.txt", package / "metadata/calibration_dataset.txt")):
            shutil.copy2(source, destination)
        shutil.copytree(smoke, package / "metadata/quantization_smoke",
                        ignore=shutil.ignore_patterns("*.rknn", "validation", "metrics"))
        subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(package / "source/code.tar.gz"), commit,
            "LICENSE", "scripts/anti_uav/dist_native", "scripts/anti_uav/rknn_yolov8_native",
            "scripts/anti_uav/build_dist_detector_on_board.sh"], cwd=ROOT, check=True)
        metrics = json.loads((a.run_dir / "evaluation/results.json").read_text())
        artifact = dict(plan, created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            training_dir=str(train_dir), delivery_dir=str(package), model_sha256=sha256(tested),
            source_pt_sha256=sha256(student), onnx_sha256=sha256(onnx),
            source_statuses=state["source_statuses"], quantization_evaluated_frames=200,
            final_binary_is_simulator_evaluated=True, board_validation_complete=False)
        dump(package / "FINAL_ARTIFACT.json", artifact)
        values = dict(TRAINING_VIDEOS=str(videos), CREATED_AT=artifact["created_at"],
            DELIVERY_DIR=str(package), TRAINING_DIR=str(train_dir), DATA_DIR=str(a.data_dir),
            MODEL_SHA256=artifact["model_sha256"], PT_SHA256=artifact["source_pt_sha256"],
            ONNX_SHA256=artifact["onnx_sha256"], COMMIT=commit,
            DATA_POLICY=json.dumps(state, ensure_ascii=False), METRICS=metric_table(metrics),
            AUGMENTATION_POLICY=augmentation_policy(manifest, state))
        document = (ROOT / "scripts/anti_uav/templates/RK3576_LABELER_HANDOVER_ZH.md").read_text()
        for key, value in values.items():
            document = document.replace("{{" + key + "}}", value)
        if "{{" in document:
            raise ValueError("Unresolved handover template field")
        (package / "DEPLOYMENT_HANDOVER_ZH.md").write_text(document)
        (package / "README.md").write_text(document)
        files = sorted(path for path in package.rglob("*") if path.is_file())
        (package / "SHA256SUMS").write_text("".join(sha256(path) + "  " + str(path.relative_to(package)) + "\n" for path in files))
        archive = package.with_name(package.name + ".tar.gz")
        if archive.exists():
            raise FileExistsError(archive)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(package, arcname=package.name)
        archive.with_name(archive.name + ".sha256").write_text(sha256(archive) + "  " + archive.name + "\n")
        status("complete", training_videos=videos, model=str(package / "models/detector_960x544_rk3576_w8a8.rknn"),
               handover=str(package / "DEPLOYMENT_HANDOVER_ZH.md"), archive=str(archive), sha256=sha256(archive))
    except BaseException as error:
        previous = json.loads((a.run_dir / "workflow_status.json").read_text())
        status("failed", failed_stage=previous.get("stage"), error=repr(error))
        raise


if __name__ == "__main__":
    main()
