#!/usr/bin/env python3
"""Paired PT/ONNX FP32 and RKNN INT8 simulator evaluation on native gray frames."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator, pool_native
from scripts.anti_uav.rknn_simulator_video_clip import decode_boxes
from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SPLITS = {
    "Video00004_test": HOLDOUT,
    "Video00009_selection_subset": Path("/mnt/andrew/anti_uav_model_refinement/data/"
        "real_gray_online_44videos_direct_20260929/train_online_gray_monitor.yaml"),
}
EXPECTED = {"Video00004_test": (2359, 448), "Video00009_selection_subset": (1421, 797)}
BACKENDS = ("pt_fp32", "onnx_fp32", "rknn_int8_simulator")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def raw_prediction(outputs, input_hw=(544, 960)):
    """Decode branch-major RK exports into the same pre-NMS tensor as PT."""
    if len(outputs) not in (9, 12):
        raise ValueError(f"Expected 9 or 12 RK-optimized outputs, got {len(outputs)}")
    branches = []
    for offset in range(0, len(outputs), 3):
        position, scores, sums = (np.asarray(x) for x in outputs[offset:offset + 3])
        if (position.ndim != 4 or position.shape[0] != 1 or position.shape[1] != 64
                or scores.shape != (1, 1, *position.shape[2:]) or sums.shape != scores.shape):
            raise ValueError("Expected batch-one NCHW bbox64/class1/score-sum1 outputs")
        if not all(np.issubdtype(x.dtype, np.floating) for x in (position, scores, sums)):
            raise ValueError("Simulator outputs must be dequantized before DFL/NMS")
        xyxy = decode_boxes(position, *input_hw).astype(np.float32)
        xywh = np.concatenate(((xyxy[:, :2] + xyxy[:, 2:]) * .5,
                                xyxy[:, 2:] - xyxy[:, :2]), axis=1)
        branches.append(np.concatenate((xywh, scores.astype(np.float32)), axis=1).reshape(1, 5, -1))
    return torch.from_numpy(np.concatenate(branches, axis=2))


def make_validator(listing, output):
    validator = PairValidator(save_dir=output, args=dict(imgsz=[544, 960], rect=False,
        batch=1, workers=4, conf=.001, iou=.45, max_det=100, half=False, plots=False))
    validator.device = torch.device("cpu")
    validator.training = False
    validator.stride = 32
    validator.data = dict(val=str(listing), names={0: "drone"}, nc=1)
    validator.init_metrics(SimpleNamespace(names={0: "drone"}))
    return validator


def report(output, result):
    protocol = result["protocol"]
    lines = ["# 44-video detector: FP32 vs INT8", "",
        f"Target: {protocol['target']}; RKNN Toolkit2 {protocol['toolkit_version']} host simulator.",
        "Not a board accuracy or FPS measurement. Tracking is excluded.",
        "Identical validation-loader RGB pixels for all backends: 960x544, padding 114.",
        "Common NMS IoU .45, max_det 100; fixed metrics conf .01/.03/.05; AP floor .001.",
        "Video00004 is held-out test; Video00009 selected checkpoints and is NOT independent test.",
        "Native frames only: zoom validation views excluded. Pooled AP is globally recomputed.",
        f"Rebuilt RKNN SHA256 equals delivered artifact: {protocol['rebuilt_matches_delivered']}.",
        "If hashes differ, this measures a rebuild with identical inputs/config, not the delivered binary.", ""]
    for split in (*SPLITS, "pooled_native"):
        lines += [f"## {split}", "", "| Backend | Conf | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for conf in (.01, .03, .05):
            for backend in BACKENDS:
                m, q = result["backends"][backend][split], f"native/c{conf:.2f}/"
                tiny = m.get(q + "long_4to8px/R")
                lines.append(f"| {backend} | {conf:.2f} | {m[q+'P']:.2%} | {m[q+'R']:.2%} | "
                    f"{m[q+'TP']} | {m[q+'FP']} | {m[q+'FN']} | {m['native/mAP50']:.2%} | "
                    f"{m['native/mAP50-95']:.2%} | {f'{tiny:.2%}' if tiny is not None else 'N/A'} |")
        lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "onnx", "rknn", "calibration", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--target", choices=("rk3576", "rk3588"), default="rk3576")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Smoke only; 0 evaluates all native frames")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def status(stage, **extra):
        value = dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=os.getpid(), **extra)
        write_json(args.output / "status.json", value)
        print(json.dumps(value), flush=True)

    runtime = None
    try:
        import cv2
        import onnxruntime as ort
        from importlib.metadata import version
        from rknn.api import RKNN

        SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False,
            hub=False, mlflow=False, neptune=False, raytune=False))
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        status("preflight")
        calibration = [s.strip() for s in args.calibration.read_text().splitlines() if s.strip()]
        if not calibration or any("Video00004" in s or "Video00009" in s for s in calibration):
            raise ValueError("Calibration must not contain evaluation videos")
        for path in calibration:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        protocol = dict(target=args.target, toolkit_version=version("rknn-toolkit2"),
            input_hw=[544, 960], padding=114, conf_floor=.001, nms_iou=.45, max_det=100,
            thresholds=[.01, .03, .05], matching_iou=.5, calibration_frames=len(calibration),
            scope="detector-only host quantization simulator", pooled_is_independent_test=False,
            smoke_limit_per_split=args.limit_per_split,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts={name: dict(path=str(getattr(args, name)), sha256=sha256(getattr(args, name)))
                       for name in ("weights", "onnx", "rknn", "calibration")}, datasets={})
        manifests = {}
        for split, config_path in SPLITS.items():
            config = yaml.safe_load(config_path.read_text())
            listing = Path(config["val"])
            native = [s.strip() for s in listing.read_text().splitlines() if s.strip() and "/zoom_val/" not in s]
            if len(native) != EXPECTED[split][0] or len(set(native)) != len(native):
                raise ValueError(f"Unexpected native coverage: {split}")
            if args.limit_per_split:
                # Spread smoke frames across the whole clip, including positives and negatives.
                native = [native[i] for i in np.linspace(0, len(native)-1,
                    min(len(native), args.limit_per_split), dtype=int)]
            listing_out = args.output / f"{split}.txt"
            listing_out.write_text("\n".join(native) + "\n")
            manifests[split] = listing_out
            protocol["datasets"][split] = dict(config=str(config_path), source_list=str(listing),
                source_list_sha256=sha256(listing), evaluated_frames=len(native),
                evaluated_list_sha256=sha256(listing_out))
        if set(manifests["Video00004_test"].read_text().splitlines()) & set(
                manifests["Video00009_selection_subset"].read_text().splitlines()):
            raise ValueError("Duplicate frames across splits")
        write_json(args.output / "protocol.json", protocol)
        status("build_quantized_simulator")
        runtime = RKNN(verbose=False)
        def require_zero(code, operation):
            if code != 0:
                raise RuntimeError(f"{operation} failed: {code}")
        require_zero(runtime.config(target_platform=args.target, mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]]), "config")
        require_zero(runtime.load_onnx(model=str(args.onnx)), "load_onnx")
        require_zero(runtime.build(do_quantization=True, dataset=str(args.calibration)), "build")
        rebuilt = args.output / "rebuilt_for_simulator.rknn"
        require_zero(runtime.export_rknn(str(rebuilt)), "export_rknn")
        protocol["rebuilt_sha256"] = sha256(rebuilt)
        protocol["rebuilt_matches_delivered"] = protocol["rebuilt_sha256"] == protocol["artifacts"]["rknn"]["sha256"]
        require_zero(runtime.init_runtime(), "init_runtime")
        write_json(args.output / "protocol.json", protocol)
        pt = YOLO(str(args.weights)).model.float().fuse().eval().to(args.device)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
        onnx = ort.InferenceSession(str(args.onnx), sess_options=opts, providers=["CPUExecutionProvider"])
        result = dict(protocol=protocol, backends={backend: {} for backend in BACKENDS},
                      raw_export_max_error=dict(box_xywh=0., class_score=0.))
        entries = {backend: [] for backend in BACKENDS}
        started = time.monotonic()
        for split, listing in manifests.items():
            validators = {backend: make_validator(listing, args.output / "metrics" / split / backend)
                          for backend in BACKENDS}
            reference = validators["pt_fp32"]
            loader = reference.get_dataloader(str(listing), 1)
            if len(loader.dataset) != protocol["datasets"][split]["evaluated_frames"]:
                raise ValueError("Missing/corrupt image: loader coverage differs from manifest")
            for index, batch in enumerate(loader):
                rgb = np.ascontiguousarray(batch["img"].numpy().transpose(0, 2, 3, 1))
                batch = reference.preprocess(batch)
                with torch.inference_mode():
                    pred = pt(batch["img"].to(args.device))
                pt_raw = (pred[0] if isinstance(pred, (tuple, list)) else pred).detach().cpu()
                ort_raw = raw_prediction(onnx.run(None, {onnx.get_inputs()[0].name: batch["img"].numpy()}))
                if pt_raw.shape != ort_raw.shape:
                    raise ValueError(f"PT/export shape mismatch: {pt_raw.shape} vs {ort_raw.shape}")
                error = (pt_raw-ort_raw).abs()
                result["raw_export_max_error"]["box_xywh"] = max(
                    result["raw_export_max_error"]["box_xywh"], float(error[:, :4].max()))
                result["raw_export_max_error"]["class_score"] = max(
                    result["raw_export_max_error"]["class_score"], float(error[:, 4:].max()))
                quantized = raw_prediction(runtime.inference(inputs=[rgb], data_format=["nhwc"]))
                for backend, raw in zip(BACKENDS, (pt_raw, ort_raw, quantized)):
                    validators[backend].update_metrics(validators[backend].postprocess(raw), batch)
                if index == 0 or (index+1) % 100 == 0:
                    status("evaluation", split=split, frames=index+1, total=len(loader.dataset),
                           elapsed_seconds=round(time.monotonic()-started, 2))
            for backend, validator in validators.items():
                metrics = {k: v for k, v in clean(validator.get_stats()).items() if k.startswith("native/")}
                coverage = metrics["native/c0.03/FRAMES"], metrics["native/c0.03/TP"]+metrics["native/c0.03/FN"]
                if not args.limit_per_split and coverage != EXPECTED[split]:
                    raise ValueError(f"Frames/GT changed for {split}: {coverage}")
                result["backends"][backend][split] = metrics
                entries[backend].append(dict(metrics=metrics, arrays=validator.metrics.native_pr_arrays,
                    scales=validator.metrics.native_scale_counts))
                np.savez_compressed(args.output / f"{split}_{backend}_pr.npz", **validator.metrics.native_pr_arrays)
            write_json(args.output / "results.json", result)
        for backend in BACKENDS:
            metrics, arrays = pool_native(entries[backend])
            result["backends"][backend]["pooled_native"] = clean(metrics)
            np.savez_compressed(args.output / f"pooled_{backend}_pr.npz", **arrays)
        for name, artifact in protocol["artifacts"].items():
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Artifact changed during evaluation: {name}")
        for manifest in protocol["datasets"].values():
            if sha256(manifest["source_list"]) != manifest["source_list_sha256"]:
                raise ValueError("Source dataset list changed during evaluation")
        write_json(args.output / "results.json", result)
        report(args.output, result)
        status("complete", report=str(args.output / "REPORT.md"), elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        if runtime is not None:
            runtime.release()


if __name__ == "__main__":
    main()
