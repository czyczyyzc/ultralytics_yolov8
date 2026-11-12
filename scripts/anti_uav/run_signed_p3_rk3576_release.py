"""Build and package the exact RK3576 INT8 Student that passed host evaluation."""

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

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = Path("/mnt/andrew/anti_uav_model_refinement/data/real_gray_yolo_lovo_positive_mixed_v1_20260828"
               "/folds/holdout_Video00004/holdout_all.yaml")
MODEL = "detector_960x544_rk3576_signed_p3_w8a8.rknn"


def read(path):
    return json.loads(Path(path).read_text())


def handover(manifest, metrics):
    lines = [
        "# RK3576 新版 P3 Student 部署交接",
        "",
        f"生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}",
        "",
        "**交付模型：97 视频修复版 P3 Student，RK3576 专用 W8A8 INT8。**",
        f"包内权重：`models/{MODEL}`；SHA256：`{manifest['rknn_sha256']}`。",
        f"交付目录：`{manifest['delivery_path']}`；压缩包：`{manifest['delivery_path']}.tar.gz`。",
        f"训练目录：`{manifest['curriculum']}`；打包源码提交：`{manifest['source_commit']}`。",
        "",
        "## 训练与精度",
        "",
        f"P3 initial 跑满 30 轮后，使用通过等训练步数验收的第 {manifest['accepted_epoch']} 轮权重冻结 P3–P5，"
        "训练 Add-on P2 Teacher（12 轮），再由该 Teacher 蒸馏 Student（15 轮）。"
        "交付的是去掉辅助 P2 和投影层后的纯 P3/P4/P5 Student best，保留 P3 有符号 DFL；无 QAT。",
        "",
        "下表为宿主机同帧评测，不是真机 FPS 或延迟。Video00004 为保留测试，Video00009 用于选权重；"
        "两段都不进入训练或量化校准。合并结果不是完全独立测试。检测口径：conf 0.03、NMS IoU 0.45、匹配 IoU 0.50，AP 分数下限 0.001。",
        "",
        "| 数据 | 模型 | Recall | Precision | FP | mAP50 | mAP50–95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, label in (("Video00004_test","Video00004"), ("Video00009_selection_subset","Video00009"),
                         ("pooled_native","合并")):
        for backend, name in (("pt_fp32","FP32"),("rknn_int8_simulator","RKNN INT8")):
            row = metrics["backends"][backend][split]
            pre = "native/c0.03/"
            lines.append(f"| {label} | {name} | {row[pre+'R']:.2%} | {row[pre+'P']:.2%} | "
                         f"{row[pre+'FP']} | {row['native/mAP50']:.2%} | {row['native/mAP50-95']:.2%} |")
    return "\n".join(lines) + """

小目标分档召回、误检、PT/ONNX 一致性和量化细节见 `metadata/quantization_results.json`；训练验收见 `metadata/p3_acceptance.json`。

## 输入输出协议

- 输入：UINT8 NHWC，宽 960、高 544、三通道相同的灰度像素。等比例双线性 letterbox，padding 114。RKNN 内部 mean 0、std 255，外部不要再除 255。
- 输出：按 P3、P4、P5 顺序，每层依次为 bbox、class、score-sum，共九路。class 和 score-sum 各 1 通道，类别 0 为航空目标。
- P3（stride 8）：bbox `1×68×68×120`，按 left/top/right/bottom 分组，每组 17 档，顺序 `−1, 0, …, 15`。反量化后对组内做 softmax，以该有符号序列求期望。
- P4（stride 16）：bbox `1×64×34×60`；P5（stride 32）：bbox `1×64×17×30`。每边仍为 `0…15` 的 16 档。
- 用实际输出 tensor 的类型、scale、zero-point 和 stride 做读取及反量化。不可以将 P3 当 64 通道，不可以把负距离截断成 0。RK3588 的二进制不能换名用于 RK3576。
- 参考运行参数：conf 0.03、NMS IoU 0.45、每帧最多 100 框。无目标面积过滤；不同场景应实测再定业务阈值。

## RK3576 板端集成

使用 aarch64、板厂匹配的 NPU 驱动和兼容的 librknnrt，部署不依赖 Python。源码包包含可读取 64/68 通道的 C++ 解码及原生跟踪链路；**必须重新编译 detector 共享库，不要只替换旧模型文件**。

```bash
sha256sum -c SHA256SUMS
PKG="$PWD"
DEPLOY="$HOME/deployments/anti_uav_signed_p3"
mkdir -p "$DEPLOY/src"
tar -xzf "$PKG/source/code.tar.gz" -C "$DEPLOY/src"
cd "$DEPLOY/src"
bash scripts/anti_uav/dist_native/build.sh "$DEPLOY"
RKNN_INCLUDE=/absolute/path/to/rknpu2/include \\
  bash scripts/anti_uav/build_dist_detector_on_board.sh "$DEPLOY/libanti_uav_detector.so"

"$DEPLOY/anti_uav_dist_native" \\
  --model "$PKG/models/detector_960x544_rk3576_signed_p3_w8a8.rknn" \\
  --detector-library "$DEPLOY/libanti_uav_detector.so" \\
  --tracker-library "$DEPLOY/libmotion_tracker.so" \\
  --gmc-library "$DEPLOY/libdist_gmc.so" \\
  --video /absolute/path/gray_input.mp4 --decoder ffmpeg --preprocess fused \\
  --detector-only --workers 2 --inflight 2 --npu-masks 0,1 \\
  --conf 0.03 --iou 0.45 --warmup 100 --output "$DEPLOY/detector_smoke"
```

依赖通常包括 `build-essential pkg-config libopencv-dev libssl-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libdrm-dev` 及对应 RKNN SDK 头文件。先用灰度视频验证 detector-only；生产彩色视频必须先转 gray3。需要跟踪时，沿用已验收的 Dist/GMC 参数并重做关联与长测，业务需要保留全部检测框时读取 observations，而不是仅输出已确认轨迹。

## 交付边界与板端验收

本包已完成 Toolkit2 2.3.2 编译和完整宿主机同帧评测；包内 RKNN 就是实际评测的二进制。尚不能据此声称 RK3576 真机已验收。请记录板端 Runtime/驱动版本、九输出形状和实际量化参数，验证典型小框位置，再测单核/双核吞吐、稳态延迟及温控。曝光起点到结果需用可确认曝光时间戳或硬件同步实测，不能用驱动取帧时刻替代。

公开数据、贴图及源码在商用交付前仍需逐项核对授权。源码沿用仓库许可证。
"""


def package(output, weights, onnx, quant, calibration, followup, curriculum):
    measured = read(quant/"results.json")
    protocol = measured["protocol"]
    evaluated = quant/"rebuilt_for_simulator.rknn"
    if (read(quant/"status.json")["stage"] != "complete" or protocol["target"] != "rk3576"
            or protocol["toolkit_version"] != "2.3.2" or protocol["input_color"] != "gray3"
            or protocol["quantized_dtype"] != "w8a8" or protocol["auto_hybrid"]
            or protocol["smoke_limit_per_split"] or sha256(evaluated) != protocol["rebuilt_sha256"]
            or protocol["artifacts"]["weights"]["sha256"] != sha256(weights)
            or protocol["artifacts"]["onnx"]["sha256"] != sha256(onnx)):
        raise ValueError("Refuse release without measured full-split W8A8 RK3576 artifact")
    errors = measured["raw_export_max_error"]
    if errors["box_xywh"] > .02 or errors["class_score"] > .0001:
        raise ValueError(f"Signed PT/ONNX output mismatch: {errors}")
    meta = read(onnx.with_suffix(".rkopt.json"))
    if not meta["signed_p3_dfl"] or [meta["output_shapes"][i][1] for i in (0,3,6)] != [68,64,64]:
        raise ValueError("Invalid release DFL layout")
    commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name("."+output.name+".staging")
    staged.mkdir(exist_ok=False)
    for directory in ("models","metadata","source"):
        (staged/directory).mkdir()
    files = {
        weights:"models/p3_student_best_fp32.pt", onnx:"models/detector_960x544_signed_p3_rkopt.onnx",
        evaluated:"models/"+MODEL, onnx.with_suffix(".rkopt.json"):"metadata/output_contract.json",
        calibration/"audit.json":"metadata/calibration_audit.json", quant/"results.json":"metadata/quantization_results.json",
        quant/"REPORT.md":"metadata/QUANTIZATION_REPORT.md", followup/"p3_acceptance.json":"metadata/p3_acceptance.json",
        curriculum/"protocol.json":"metadata/training_protocol.json", curriculum/"audit/teacher_frozen_p3.json":"metadata/teacher_frozen_p3.json",
        curriculum/"audit/student_exports.json":"metadata/student_exports.json",
        curriculum/"comparison_video00009.json":"metadata/fp32_curriculum_comparison.json",
    }
    for src,dst in files.items():
        shutil.copy2(src,staged/dst)
    subprocess.run(["git","archive","--format=tar.gz","--output",str(staged/"source/code.tar.gz"),commit,
        "LICENSE","scripts/anti_uav/dist_native","scripts/anti_uav/build_dist_detector_on_board.sh",
        "scripts/anti_uav/rknn_yolov8_native"],cwd=ROOT,check=True)
    gate = read(followup/"p3_acceptance.json")
    manifest = dict(delivery_path=str(output), curriculum=str(curriculum), accepted_epoch=gate["epoch"],
        training_videos=97, target="rk3576", quantization="W8A8 normal/channel PTQ", toolkit="2.3.2",
        rknn_sha256=sha256(staged/"models"/MODEL), source_commit=commit, host_accuracy_verified=True,
        board_verified=False, signed_p3_dfl=True, teacher_weights_sha256=sha256(curriculum/"teacher/p2p3/weights/best.pt"))
    dump(staged/"FINAL_ARTIFACT.json",manifest)
    (staged/"DEPLOYMENT_HANDOVER_ZH.md").write_text(handover(manifest,measured))
    hashes = [f"{sha256(p)}  {p.relative_to(staged)}" for p in sorted(staged.rglob("*")) if p.is_file()]
    (staged/"SHA256SUMS").write_text("\n".join(hashes)+"\n")
    archive = output.with_suffix(".tar.gz")
    archive_tmp = archive.with_name("."+archive.name+".tmp")
    with tarfile.open(archive_tmp,"w:gz") as stream:
        stream.add(staged,arcname=output.name)
    if output.exists() or archive.exists():
        raise FileExistsError("Refuse to overwrite an existing delivery")
    staged.rename(output)
    archive_tmp.rename(archive)
    archive.with_suffix(".gz.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
    subprocess.run(["sha256sum","-c","SHA256SUMS"],cwd=output,check=True)
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("followup","data","work-dir","delivery"):
        p.add_argument(f"--{name}",required=True,type=lambda s: Path(s).resolve())
    a = p.parse_args()
    a.work_dir.mkdir(parents=True,exist_ok=True)
    lock = (a.work_dir/"release.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs = a.work_dir/"logs"
    logs.mkdir(exist_ok=True)

    def status(stage,**more):
        dump(a.work_dir/"status.json",dict(stage=stage,pid=os.getpid(),updated=time.time(),**more))

    def command(name,module,*args):
        status(name)
        with (logs/f"{name}.log").open("x") as stream:
            subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,check=True)

    try:
        calib = a.work_dir/"calibration"
        holdout = yaml.safe_load(HOLDOUT.read_text())
        if not (calib/"audit.json").is_file():
            command("prepare_calibration","scripts.anti_uav.prepare_stratified_rknn_calibration",
                "--data",a.data,"--output-dir",calib,"--samples",384,
                "--exclude-token","Video00004","Video00009","--exclude-list",holdout["val"])
        audit = read(calib/"audit.json")
        if (audit["data"] != str(a.data) or audit["input_color"] != "gray3" or audit["samples"] != 384
                or sha256(audit["train_list"]) != audit["train_list_sha256"]):
            raise ValueError("Calibration is not the pinned gray3 training-only set")
        while True:
            follow = read(a.followup/"supervisor_status.json")
            if follow["stage"] in {"failed","p3_gate_not_met"}:
                status("awaiting_upstream_resolution", upstream=follow["stage"], detail=follow)
                return
            if follow["stage"] == "curriculum_complete":
                break
            status("waiting_for_teacher_student", upstream=follow["stage"])
            time.sleep(60)
        curriculum = Path(follow["curriculum"])
        weights = curriculum/"p3_student_distilled_best_fp32.pt"
        if (read(curriculum/"status.json")["stage"] != "complete"
                or read(a.followup/"p3_acceptance.json")["data_sha256"] != sha256(a.data)
                or not read(curriculum/"audit/teacher_frozen_p3.json")["bit_exact"]):
            raise ValueError("Student provenance or frozen Teacher verification incomplete")
        onnx = a.work_dir/"signed_student_rkopt.onnx"
        command("export_onnx","scripts.anti_uav.export_detector_rkopt_onnx",
            "--weights",weights,"--output",onnx,"--imgsz","544,960","--device","cpu")
        # Retain the simulator-built artifact, even if bytewise different from this reference build.
        rknn = a.work_dir/"reference_rk3576_w8a8.rknn"
        command("compile_rk3576","scripts.anti_uav.build_rknn","--onnx",onnx,"--output",rknn,
            "--target","rk3576","--quantize","--dataset",calib/"dataset.txt")
        quant = a.work_dir/"quantization"
        command("evaluate_int8","scripts.anti_uav.compare_gray_rknn_quantization",
            "--weights",weights,"--onnx",onnx,"--rknn",rknn,"--calibration",calib/"dataset.txt",
            "--target","rk3576","--device","cpu","--input-color","gray3","--video00009-data",a.data,"--output",quant)
        status("package_delivery")
        archive = package(a.delivery,weights,onnx,quant,calib,a.followup,curriculum)
        status("complete",archive=str(archive),sha256=sha256(archive),handover=str(a.delivery/"DEPLOYMENT_HANDOVER_ZH.md"))
    except BaseException as error:
        status("failed",error=repr(error))
        raise


if __name__ == "__main__":
    main()
