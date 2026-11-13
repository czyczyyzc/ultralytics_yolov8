"""Train only a Frozen-P3 Add-on P2 Teacher from a pinned expanded-data checkpoint."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess

import torch
import yaml

from scripts.anti_uav.evaluate_p3_equal_fp import evaluate
from scripts.anti_uav.frozen_p3_addon_p2_trainer import add_on_parameter_ids
from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
from scripts.anti_uav.prepare_labeler_training_expansion import dump
from scripts.anti_uav.regression_repair_trainers import RegressionRepairAddOnTrainer
from scripts.anti_uav.run_approved_gray_fixedquota_refresh import choose_shared_gpu
from scripts.anti_uav.run_p3_independent_curriculum_training import (
    SEED, add_status_callback, audit_label_caches, common_train, sha256, stage_yaml, write_status,
)
from scripts.anti_uav.train_frozen_p3_addon_p2 import DEFAULT_CFG, initialize_addon_model, verify_legacy_outputs
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.torch_utils import de_parallel, init_seeds


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("foundation-launch", "snapshot", "holdout", "run-dir"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--signed-p2", action="store_true", help="Add the negative P2 bin and off-grid fallback")
    parser.add_argument("--p2-control-run", type=Path, help="Include the completed 64-channel P2 Teacher if available")
    args = parser.parse_args()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "teacher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "status.json").exists():
        raise FileExistsError("Refuse to overwrite an existing Teacher experiment")

    try:
        write_status(run, "check_teacher_inputs")
        receipt = read(args.snapshot)
        upstream = read(args.foundation_launch / "launch_contract.json")
        protocol = read(args.foundation_launch / "candidate/protocol.json")
        epoch = receipt["epoch"]
        data = Path(upstream["data"])
        source = yaml.safe_load(data.read_text())
        manifest = read(data.parent / "manifest.json")
        videos = set(manifest["train_video_hashes"])
        checkpoint = args.snapshot.parent / "p3_initial_last.pt"
        if (receipt["checkpoint_kind"] != "last" or receipt["stage"] != "p3_initial"
                or receipt["data"] != str(data) or protocol["data"] != str(data)
                or receipt["train_commit"] != protocol["git_commit"]
                or receipt["checkpoint_sha256"] != args.checkpoint_sha256
                or sha256(checkpoint) != args.checkpoint_sha256 or sha256(data) != upstream["data_sha256"]
                or not 1 <= epoch <= protocol["epochs"]["foundation"]
                or protocol["training_videos"] != len(videos) or len(videos) != upstream["training_videos"]
                or {manifest["validation_sha256"], manifest["test_sha256"]} & videos
                or protocol["p3_signed_dfl"]["bins"] != [-1, 15]
                or source["input_color"] != "gray3" or source["visible_box_guard"] != "affine_keep_subpixel_v2"
                or source["tiny_sampling"]["batch_quotas"] != upstream["batch_quotas"]):
            raise ValueError("Foundation snapshot or frozen dataset contract changed")
        foundation = run / "p3_initial_pinned.pt"
        shutil.copy2(checkpoint, foundation)
        if sha256(foundation) != args.checkpoint_sha256:
            raise ValueError("Pinned foundation copy does not match its receipt")

        SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                             mlflow=False, neptune=False, raytune=False))
        torch.set_num_threads(4)
        init_seeds(SEED, deterministic=True)
        args.batch, args.workers = protocol["batch"], 16
        epochs = protocol["epochs"]["teacher"]
        audit = run / "audit"
        audit.mkdir()
        source["trusted_label_caches"] = audit_label_caches(source)
        if source["trusted_label_caches"] != protocol["trusted_label_caches"]:
            raise ValueError("Train or validation cache differs from the foundation")
        teacher_data = stage_yaml(source, run / "teacher.yaml", epoch,
                                  protocol["augmentation"]["online_replacement_probability"])
        dump(run / "protocol.json", dict(
            stage="parallel_teacher_only", training_videos=len(videos), epochs=epochs, foundation_epoch=epoch,
            foundation=str(foundation), foundation_sha256=args.checkpoint_sha256, data=str(data),
            data_sha256=upstream["data_sha256"], batch_quotas=upstream["batch_quotas"],
            batches_per_epoch=upstream["batches_per_epoch"], student_queued=False, signed_p2=args.signed_p2,
            p2_regression_channels=68 if args.signed_p2 else 64,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))

        write_status(run, "initialize_teacher", foundation_epoch=epoch)
        teacher, initialization = initialize_addon_model(
            foundation, DEFAULT_CFG, run / "teacher_initialized.pt", signed_p2=args.signed_p2)
        dump(audit / "teacher_initialization.json", initialization)
        trainer_class = RegressionRepairAddOnTrainer
        trainer_class.repair_options = protocol["regression_repair"]["options"]
        trainer_class.validator_class = FixedShapeGrayValidator
        trainer_class.tiny_multiplier = 1.75

        def check_training_start(trainer):
            model = de_parallel(trainer.model)
            trainable = {id(p) for p in model.parameters() if p.requires_grad}
            expected = add_on_parameter_ids(model)
            batches = len(trainer.train_loader)
            if trainable != expected or batches != upstream["batches_per_epoch"]:
                raise ValueError("Trainable parameter scope or batch count changed")
            if hasattr(model.model[-1], "p2_negative_bin") != args.signed_p2:
                raise ValueError("Teacher P2 head does not match the requested variant")
            names = [name for name, p in model.named_parameters() if p.requires_grad]
            dump(audit / "trainable_scope.json", dict(p2_only=True, trainable_tensors=names, batches=batches))
            write_status(run, "train_teacher", epoch=0, device=args.device, batches_per_epoch=batches)

        usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True)
        gpu = choose_shared_gpu(usage, args.gpus, False)
        if gpu is None:
            raise RuntimeError("No permitted idle GPU; not sharing an occupied card")
        args.device = str(gpu)
        add_status_callback(teacher, run, "train_teacher")
        teacher.add_callback("on_train_start", check_training_start)
        write_status(run, "prepare_teacher_training", device=args.device)
        teacher.train(trainer=trainer_class, project=str(run / "teacher"), name="p2p3",
                      lr0=.001, warmup_epochs=1, **common_train(teacher_data, args, epochs))

        trained = read(run / "status.json")
        if trained.get("epoch") != epochs:
            raise RuntimeError("Teacher did not finish the scheduled epochs")
        teacher_best = run / "teacher/p2p3/weights/best.pt"
        frozen = verify_legacy_outputs(YOLO(str(foundation)).model, YOLO(str(teacher_best)).model)
        dump(audit / "teacher_frozen_p3.json", frozen)
        write_status(run, "evaluate_teacher_equal_fp", device=args.device, epoch=epochs)
        arms = {"p3_initial": foundation, "p2p3_teacher": teacher_best}
        if args.p2_control_run:
            control = read(args.p2_control_run / "protocol.json")
            current = read(run / "protocol.json")
            keys = ("foundation_sha256", "data_sha256", "epochs", "batch_quotas", "batches_per_epoch")
            if control.get("signed_p2") or any(control[k] != current[k] for k in keys):
                raise ValueError("The P2 control does not match the experiment")
            control_status = read(args.p2_control_run / "status.json")
            if control_status["stage"] in {"teacher_complete", "evaluate_teacher_equal_fp"}:
                control_weights = args.p2_control_run / "teacher/p2p3/weights/best.pt"
                verify_legacy_outputs(YOLO(str(foundation)).model, YOLO(str(control_weights)).model)
                arms["p2p3_teacher_control"] = control_weights
        evaluate(arms, {"Video00009": data, "Video00004": args.holdout}, args.device, run / "comparison")
        write_status(run, "teacher_complete", device=args.device, epoch=epochs, teacher_best=str(teacher_best),
                     comparison=str(run / "comparison/comparison.json"), student_queued=False)
    except BaseException as error:
        write_status(run, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
