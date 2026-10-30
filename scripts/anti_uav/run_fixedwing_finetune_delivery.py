#!/usr/bin/env python3
"""Wait for the specified urgent fine-tune, evaluate, then package RK3576 INT8."""

import argparse
import csv
import fcntl
import json
import math
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
from scripts.anti_uav.prepare_fixedwing_finetune import read_pinned_cache, write_split
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def completed_training(run, expected_epochs=8):
    state = json.loads((run/"status.json").read_text())
    if state["stage"] == "failed":
        raise RuntimeError(f"Fine-tuning failed: {state}")
    if state["stage"] != "complete":
        return None
    with (run/"training/p3/results.csv").open() as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    if [int(x["epoch"]) for x in rows] != list(range(1, expected_epochs+1)):
        raise ValueError("Refuse to deliver a partial fine-tune")
    comparison = json.loads((run/"comparison.json").read_text())
    candidate = run/"p3_student_fixedwing_best_fp32.pt"
    if comparison["candidate"] != str(candidate) or sha256(candidate) != comparison["sha256"]:
        raise ValueError("Fine-tuned checkpoint differs from the final evaluated checkpoint")
    for key in ("native/c0.03/R", "native/c0.03/P", "adaptation_fit/c0.03/R", "adaptation_fit/c0.03/P"):
        value = comparison["final"][key]
        if value is None or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Final detector metrics are missing or invalid")
    return comparison


def metric_row(name, metrics):
    q = "native/c0.03/"
    return (f"| {name} | {metrics[q+'R']:.2%} | {metrics[q+'P']:.2%} | {int(metrics[q+'TP'])} | "
            f"{int(metrics[q+'FP'])} | {int(metrics[q+'FN'])} | {metrics['native/mAP50']:.2%} | "
            f"{metrics['native/mAP50-95']:.2%} |")


def metric_table(records):
    return "\n".join(["| 版本 | Recall | Precision | TP | FP | FN | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]+
        [metric_row(name, metrics) for name, metrics in records.items()])


def release_note(comparison):
    if comparison["ready_for_quantization_review"]:
        return "微调后的新增场景拟合 Recall 提高，且通过本轮旧场景 Recall 保留检查。仍需在独立固定翼视频和目标板上验收。"
    return ("本模型面向新增固定翼场景适配，未同时通过全部旧场景 Recall 保留及新增拟合提升条件。"
            "请先核对以下取舍，不应作为原通用模型的无条件替代；独立固定翼精度和板端性能仍待验收。")


def retained_release_eligible(old, new):
    return (new["pooled_native"]["native/c0.03/R"] >= old["pooled_native"]["native/c0.03/R"]-.03
        and new["Video00004_test"]["native/c0.03/R"] >= old["Video00004_test"]["native/c0.03/R"]-.05)


def render_handover(values, root=ROOT, method_id=None):
    name = ("RK3576_DISTILLED_FINETUNE_HANDOVER_ZH.md" if method_id == "continued_frozen_teacher_student_v1"
            else "RK3576_FIXEDWING_FINETUNE_HANDOVER_ZH.md")
    header = (root/"scripts/anti_uav/templates"/name).read_text()
    shared = (root/"scripts/anti_uav/templates/RK3576_LABELER_HANDOVER_ZH.md").read_text()
    marker = "## 5. 模型与图像协议"
    if shared.count(marker) != 1:
        raise ValueError("Shared deployment protocol section changed")
    document = header+"\n"+marker+shared.split(marker)[1]
    for key, value in values.items():
        document = document.replace("{{"+key+"}}", value)
    if "{{" in document:
        raise ValueError("Unresolved delivery template variable")
    return document


def make_data(run, output):
    import yaml
    source = yaml.safe_load((run/"data/train.yaml").read_text())
    audit = json.loads((run/"data/manifest.json").read_text())
    train = read_pinned_cache(source["train"], source)
    validation = read_pinned_cache(source["val"], source)
    roots = set(audit["adaptation_roots"])
    new_rows = [x for x in train["labels"] if str(Path(x["im_file"]).parent) in roots]
    monitor = [x for x in validation["labels"] if str(Path(x["im_file"]).parent) in roots]
    native = [x for x in validation["labels"] if str(Path(x["im_file"]).parent) not in roots]
    expected = audit["new_positive"]+audit["new_reviewed_negative"]
    if len(new_rows) != expected or len({x["im_file"] for x in new_rows}) != expected:
        raise ValueError("New-video full-fit coverage changed")
    output.mkdir()
    def config(name, rows):
        listing = output/(name+".txt")
        pin = write_split(listing, rows, train["version"], audit["source_snapshot_sha256"])
        path = output/(name+".yaml")
        value = dict(path=str(output), train=source["train"], val=str(listing), names=source["names"],
                     trusted_label_caches={str(listing): pin})
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        return path
    full_fit, monitor_data, native_data = (config(name, rows) for name, rows in (
        ("new21_training_fit_full", new_rows), ("new21_training_fit_monitor", monitor),
        ("native_validation", native)))
    # Never calibrate on the new-video frames used for the paired INT8 check.
    excluded = {x["im_file"] for x in validation["labels"]}
    calibration_rows = [x for x in train["labels"] if x["im_file"] not in excluded]
    listing = output/"calibration_training.txt"
    pin = write_split(listing, calibration_rows, train["version"], audit["source_snapshot_sha256"])
    calibration_data = output/"calibration.yaml"
    calibration_data.write_text(yaml.safe_dump(dict(train=str(listing), val=source["val"],
        names=source["names"], trusted_label_caches={str(listing): pin}), sort_keys=False))
    dump(output/"coverage.json", dict(full_new_frames=len(new_rows), monitor_frames=len(monitor),
        calibration_excludes_monitor=True, new_fit_is_independent_test=False,
        full_new_gt=sum(len(x["cls"]) for x in new_rows)))
    return full_fit, monitor_data, native_data, calibration_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--delivery-dir", type=Path, required=True)
    p.add_argument("--toolkit-python", type=Path, required=True)
    p.add_argument("--training-pid", type=int, required=True)
    p.add_argument("--device", default="2")
    a = p.parse_args()
    a.training_run, a.delivery_dir = a.training_run.resolve(), a.delivery_dir.resolve()
    if not a.device.isdecimal() or a.delivery_dir.exists():
        raise ValueError("Use a fresh package and a single physical GPU")
    work = a.training_run/"delivery_pipeline"
    work.mkdir()
    (work/"logs").mkdir()
    lock = (work/"pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="4", MPLBACKEND="Agg", WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    def status(stage, **extra):
        dump(work/"workflow_status.json", dict(stage=stage, pid=os.getpid(),
            time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra))
    def execute(stage, command):
        status(stage, command=command)
        with (work/"logs"/(stage+".log")).open("a") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        status("waiting_for_8epoch_finetune")
        initial_protocol = json.loads((a.training_run/"protocol.json").read_text())
        continued = initial_protocol.get("method_id") == "continued_frozen_teacher_student_v1"
        baseline = Path(initial_protocol["initial_student"])
        if initial_protocol["epochs"] != 8 or sha256(baseline) != initial_protocol["initial_student_sha256"]:
            raise ValueError("Unexpected fine-tune protocol or changed initial model")
        pins = {str(path): sha256(path) for path in (a.training_run/"protocol.json",
                a.training_run/"data/train.yaml", a.training_run/"data/manifest.json", baseline)}
        dump(work/"protocol.json", dict(training_run=str(a.training_run), training_pid=a.training_pid,
            input_pins=pins, target="rk3576", quantization="W8A8 normal/channel PTQ", qat=False,
            new_video_fit_is_independent=False, board_validation_complete=False,
            simulator_frames_per_split=100, simulator_splits=["Video00004", "Video00009", "new21_fit_monitor"]))
        deadline = time.monotonic()+24*3600
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Fine-tuning did not finish within 24 hours")
            comparison = completed_training(a.training_run)
            proc = Path(f"/proc/{a.training_pid}/cmdline")
            active = proc.exists() and str(a.training_run).encode() in proc.read_bytes()
            if comparison is not None and not active:
                break
            if comparison is None and not active:
                raise RuntimeError("Training process exited without a completed evaluated candidate")
            time.sleep(30)
        if any(sha256(path) != digest for path, digest in pins.items()):
            raise ValueError("Pinned fine-tune inputs changed while waiting")
        if continued and not comparison["ready_for_quantization_review"]:
            status("quality_review_required", reason="Restored distillation did not pass hard old-scene retention and new-fit improvement gates; no replacement delivery produced",
                   comparison=str(a.training_run/"comparison.json"))
            return
        status("waiting_for_gpu_release")
        deadline = time.monotonic()+1800
        while True:
            used = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
                "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            memory, utilization = map(int, used.strip().split(','))
            if memory <= 256 and utilization <= 5:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("GPU taken by another process; no preemption attempted")
            time.sleep(30)
        full_fit, monitor_data, native_data, calibration_data = make_data(a.training_run, work/"data")
        student = Path(comparison["candidate"])
        models = {"old52_student": str(baseline), "fixedwing_finetuned_student": str(student)}
        dump(work/"models.json", models)
        execute("10_retained_pair_evaluation", [sys.executable, "-u",
            str(ROOT/"scripts/anti_uav/evaluate_expanded_checkpoint_pair.py"), "--models", str(work/"models.json"),
            "--validation-data", str(native_data), "--output", str(work/"evaluation"), "--device", a.device])
        retained_metrics = json.loads((work/"evaluation/results.json").read_text())
        if continued:
            old = retained_metrics["old52_student"]["splits"]
            new = retained_metrics["fixedwing_finetuned_student"]["splits"]
            if not retained_release_eligible(old, new):
                status("quality_review_required", reason="Final retained-set acceptance failed; do not silently replace the old general model",
                       comparison=str(work/"evaluation/results.json"))
                return
        execute("11_new_video_full_fit", [sys.executable, "-u", str(Path(__file__).resolve()),
            "--evaluate-fit", str(full_fit), "--models", str(work/"models.json"),
            "--output", str(work/"new21_full_fit.json"), "--device", a.device])
        export = work/"export"
        export.mkdir()
        onnx = export/"detector_960x544_rkopt.onnx"
        execute("20_onnx", [sys.executable, str(ROOT/"scripts/anti_uav/export_detector_rkopt_onnx.py"),
            "--weights", str(student), "--imgsz", "544,960", "--device", "cpu", "--output", str(onnx)])
        shapes = json.loads(onnx.with_suffix(".rkopt.json").read_text())["validation"]
        if shapes["branch_count"] != 3 or shapes["output_count"] != 9:
            raise ValueError("Export unexpectedly contains P2")
        calibration = export/"calibration"
        execute("21_calibration", [sys.executable, str(ROOT/"scripts/anti_uav/prepare_stratified_rknn_calibration.py"),
            "--data", str(calibration_data), "--output-dir", str(calibration), "--samples", "384",
            "--exclude-token", "Video00004", "Video00009", "zoom_val", "zoom_train"])
        reference = export/"reference_rk3576_w8a8.rknn"
        execute("22_rk3576_build", [str(a.toolkit_python), str(ROOT/"scripts/anti_uav/build_rknn.py"),
            "--onnx", str(onnx), "--output", str(reference), "--target", "rk3576", "--quantize",
            "--dataset", str(calibration/"dataset.txt"), "--quantized-dtype", "w8a8",
            "--quantized-algorithm", "normal", "--quantized-method", "channel"])
        smoke = export/"quantization_check"
        execute("23_paired_quantization_check", [str(a.toolkit_python), "-u",
            str(ROOT/"scripts/anti_uav/compare_gray_rknn_quantization.py"), "--weights", str(student),
            "--onnx", str(onnx), "--rknn", str(reference), "--calibration", str(calibration/"dataset.txt"),
            "--target", "rk3576", "--output", str(smoke), "--device", "cuda:"+a.device,
            "--limit-per-split", "100", "--adaptation-data", str(monitor_data)])
        quantization = json.loads((smoke/"results.json").read_text())
        tested = smoke/"rebuilt_for_simulator.rknn"
        if (json.loads((smoke/"status.json").read_text())["stage"] != "complete"
                or sha256(tested) != quantization["protocol"]["rebuilt_sha256"]
                or quantization["protocol"]["target"] != "rk3576"):
            raise ValueError("Cannot identify the actual tested RK3576 binary")
        if (sum(x["evaluated_frames"] for x in quantization["protocol"]["datasets"].values()) != 300
                or quantization["protocol"]["artifacts"]["weights"]["sha256"] != sha256(student)):
            raise ValueError("Quantization check used a different checkpoint or frame coverage")
        execute("24_load_check", [str(a.toolkit_python), "-c",
            "from rknn.api import RKNN; import sys; r=RKNN(); v=r.load_rknn(sys.argv[1]); print('load_rknn_ret=',v); r.release(); assert v==0", str(tested)])
        status("30_packaging")
        package = a.delivery_dir
        package.mkdir(parents=True)
        for name in ("models", "metadata", "source"):
            (package/name).mkdir()
        sources = {"models/p3_student_best_fp32.pt": student,
            "models/detector_960x544_rkopt.onnx": onnx,
            "models/detector_960x544_rk3576_w8a8.rknn": tested,
            "metadata/detector.rkopt.json": onnx.with_suffix(".rkopt.json"),
            "metadata/training_protocol.json": a.training_run/"protocol.json",
            "metadata/training_results.csv": a.training_run/"training/p3/results.csv",
            "metadata/sampler_audit.json": a.training_run/"training/p3/sampler_audit.json",
            "metadata/adaptation_monitor_comparison.json": a.training_run/"comparison.json",
            "metadata/data_manifest.json": a.training_run/"data/manifest.json",
            "metadata/fp32_comparison.json": work/"evaluation/results.json",
            "metadata/new21_full_fit.json": work/"new21_full_fit.json",
            "metadata/evaluation_protocol.json": work/"evaluation/protocol.json",
            "metadata/coverage.json": work/"data/coverage.json",
            "metadata/calibration_audit.json": calibration/"audit.json",
            "metadata/calibration_dataset.txt": calibration/"dataset.txt",
            "metadata/workflow_protocol.json": work/"protocol.json"}
        for relative, path in sources.items():
            shutil.copy2(path, package/relative)
        if continued:
            for name in ("initial_state_verification", "student_reconstruction_probe", "teacher_frozen_p3",
                         "teacher_checkpoint_selection", "student_checkpoint_selection"):
                path = a.training_run/"audit"/(name+".json")
                if path.exists():
                    shutil.copy2(path, package/"metadata"/path.name)
        shutil.copytree(smoke, package/"metadata/quantization_check",
            ignore=shutil.ignore_patterns("*.rknn", "metrics"))
        shutil.copytree(work/"logs", package/"metadata/logs")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        subprocess.run(["git", "archive", "--format=tar.gz", "--output="+str(package/"source/code.tar.gz"),
            commit, "LICENSE", "scripts/anti_uav/dist_native", "scripts/anti_uav/rknn_yolov8_native",
            "scripts/anti_uav/build_dist_detector_on_board.sh"], cwd=ROOT, check=True)
        retained = json.loads((work/"evaluation/results.json").read_text())
        fit = json.loads((work/"new21_full_fit.json").read_text())
        values = dict(CREATED_AT=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), DELIVERY_DIR=str(package),
            TRAINING_DIR=str(a.training_run), DATA_DIR=str(a.training_run/"data"), COMMIT=commit,
            MODEL_SHA256=sha256(tested), PT_SHA256=sha256(student), ONNX_SHA256=sha256(onnx),
            RETAINED_METRICS=metric_table({k: v["splits"]["pooled_native"] for k, v in retained.items()}),
            FIT_METRICS=metric_table({k: v["metrics"] for k, v in fit.items()}), RELEASE_NOTE=release_note(comparison),
            QUANT_NATIVE_METRICS=metric_table({k: v["pooled_native"] for k, v in quantization["backends"].items()}),
            QUANT_FIT_METRICS=metric_table({k: v["new21_training_fit_monitor"] for k, v in quantization["backends"].items()}))
        document = render_handover(values, method_id=initial_protocol.get("method_id"))
        for name in ("DEPLOYMENT_HANDOVER_ZH.md", "README.md"):
            (package/name).write_text(document)
        artifact = dict(training_run=str(a.training_run), delivery_dir=str(package),
            deployment_target="rk3576", quantization="W8A8 INT8 PTQ", source_model_sha256=sha256(student),
            model_sha256=sha256(tested), input_hw=[544, 960], final_binary_is_simulator_evaluated=True,
            quantization_evaluated_frames=300, board_validation_complete=False,
            old_recall_retention_and_fit_gate_passed=comparison["ready_for_quantization_review"], git_commit=commit)
        dump(package/"FINAL_ARTIFACT.json", artifact)
        files = sorted(p for p in package.rglob("*") if p.is_file())
        (package/"SHA256SUMS").write_text("".join(sha256(p)+"  "+str(p.relative_to(package))+"\n" for p in files))
        archive = package.with_name(package.name+".tar.gz")
        if archive.exists():
            raise FileExistsError(archive)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(package, arcname=package.name)
        archive.with_name(archive.name+".sha256").write_text(sha256(archive)+"  "+archive.name+"\n")
        status("complete", model=str(package/"models/detector_960x544_rk3576_w8a8.rknn"),
            handover=str(package/"DEPLOYMENT_HANDOVER_ZH.md"), archive=str(archive), sha256=sha256(archive),
            old_recall_retention_and_fit_gate_passed=comparison["ready_for_quantization_review"])
    except BaseException as error:
        status("failed", error=repr(error))
        raise


def evaluate_fit():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluate-fit", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", required=True)
    a = p.parse_args()
    import torch
    import yaml
    from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False,
        hub=False, mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = yaml.safe_load(a.evaluate_fit.read_text())
    cache = read_pinned_cache(data["val"], data)
    expected = len(cache["labels"]), sum(len(x["cls"]) for x in cache["labels"])
    records = {}
    for name, checkpoint in json.loads(a.models.read_text()).items():
        digest = sha256(checkpoint)
        model = YOLO(checkpoint)
        metrics = model.val(data=str(a.evaluate_fit), validator=FixedShapeGrayValidator,
            imgsz=[544, 960], rect=False, device=a.device, batch=64, workers=12, conf=.001,
            iou=.45, max_det=100, half=False, plots=False, verbose=False,
            project=str(a.output.parent/"fit_validation"), name=name)
        m = {k: None if isinstance(v, float) and not math.isfinite(v) else v
             for k, v in metrics.gray_selection.items() if k.startswith("native/")}
        if (m["native/c0.03/FRAMES"], m["native/c0.03/TP"]+m["native/c0.03/FN"]) != expected:
            raise ValueError("New-video full-fit coverage differs from snapshot")
        if sha256(checkpoint) != digest:
            raise ValueError("Checkpoint changed during evaluation")
        records[name] = dict(weights=checkpoint, sha256=digest, metrics=m,
            scope="all newly labeled training frames; NOT an independent test")
        del model
        torch.cuda.empty_cache()
    dump(a.output, records)


if __name__ == "__main__":
    evaluate_fit() if "--evaluate-fit" in sys.argv else main()
