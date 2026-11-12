"""Queue a foundation-only A/B; wait for frozen data QA and an idle GPU."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
from scripts.anti_uav.run_approved_gray_fixedquota_refresh import choose_shared_gpu, training_command

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "preparation", "run-dir", "video00004"):
        p.add_argument(f"--{name}", required=True, type=lambda path: Path(path).resolve())
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--data-root", type=Path)
    p.add_argument("--video-root", type=Path)
    p.add_argument("--frame-assets", type=Path)
    p.add_argument("--gray-batch-quotas", type=int, nargs=3)
    p.add_argument("--automatic-cache-qa", action="store_true", help="Accept verified composition samples without a separate visual sign-off")
    a = p.parse_args()
    if a.gray_batch_quotas and not (a.data_root and a.video_root and a.frame_assets):
        p.error("Quota recovery requires --data-root, --video-root and --frame-assets")
    a.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.run_dir/"ab.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = a.run_dir/"ab_logs"
    logs.mkdir(exist_ok=True)

    def status(stage, **more):
        dump(a.run_dir/"ab_status.json", dict(stage=stage, pid=os.getpid(), updated=time.time(), **more))

    def command(stage, args, gpu):
        status(stage, gpu=gpu, training_videos=prepared["training_videos"], command=list(map(str,args)))
        with (logs/f"{stage}.log").open("x") as stream:
            subprocess.run(list(map(str,args)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def idle_gpu(only=None):
        while True:
            usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True)
            gpu = choose_shared_gpu(usage, [only] if only is not None else a.gpus, False)
            if gpu is not None:
                return gpu
            status("waiting_for_idle_gpu", same_gpu_for_both_arms=only, usage=usage)
            time.sleep(60)

    try:
        while not (a.preparation/"preparation.json").is_file():
            preparation = read(a.preparation/"refresh_status.json")
            if preparation["stage"] == "failed":
                if a.gray_batch_quotas and "Native quota exceeds its bounded replay limit" in preparation["error"]:
                    contract = read(a.preparation/"refresh_contract.json")
                    frozen = read(a.preparation/"audit/requested_scope.json")
                    cache = a.data_root.resolve()/"backgrounds"
                    if read(cache/"summary.json")["stage"] != "complete":
                        raise ValueError("Quota-only recovery requires the original frozen cache")
                    status("configuring_common_gray_quotas", gray_batch_quotas=a.gray_batch_quotas)
                    with (logs/"quota_recovery.log").open("x") as stream:
                        subprocess.run(list(map(str, [sys.executable,"-u","-m","scripts.anti_uav.run_approved_gray_fixedquota_refresh",
                            "--source-run",a.source_run,"--native-control-run",a.source_run,
                            "--database",frozen["source_database"],"--video-root",a.video_root,
                            "--previous-cache",contract["previous_cache"],"--data-output",a.data_root.resolve()/"dataset",
                            "--cache-output",cache,"--run-dir",a.preparation,"--all-approved-and-submitted",
                            "--approved-frame-assets",a.frame_assets,"--visible-box-guard","affine_keep_subpixel_v2",
                            "--gray-batch-quotas",*a.gray_batch_quotas,"--prepare-only","--resume-preparation"])),
                            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    continue
                raise ValueError(f'Dataset preparation failed: {preparation["error"]}')
            status("waiting_for_frozen_data", preparation_stage=preparation["stage"])
            time.sleep(30)
        prepared = read(a.preparation/"preparation.json")
        if not prepared["complete"]:
            raise ValueError("Data preparation incomplete")
        audit = a.preparation/"audit"
        while prepared["visual_review_pending"]:
            if a.automatic_cache_qa:
                qa = audit/"composition_acceptance.json"
                composition = read(qa)
                if (composition["stage"] != "complete" or composition["rendered"] < 1
                        or len(composition["original_gt_previews"]) != len(prepared["requested_sha256"])):
                    raise ValueError("Automated frozen composition/GT coverage is incomplete")
                dump(a.run_dir/"cache_qa.json", dict(method="automated_frozen_sample_checks_not_human_review",
                    composition_acceptance_sha256=sha256(qa), verified_compositions=composition["rendered"],
                    gt_preview_count=len(composition["original_gt_previews"])))
                break
            path = audit/"visual_review.json"
            if path.is_file():
                review = read(path)
                if (review.get("accepted") is not True
                        or review.get("composition_acceptance_sha256") != sha256(audit/"composition_acceptance.json")):
                    raise ValueError("Visual review rejected or did not match the frozen sample set")
                break
            status("waiting_for_visual_review", training_videos=prepared["training_videos"],
                   previews=str(a.preparation/"replacement_QA"))
            time.sleep(30)
        data, foundation = Path(prepared["data"]), Path(prepared["foundation"])
        protocol = read(a.source_run/"protocol.json")
        manifest = read(data.parent/"manifest.json")
        for path, digest in ((data, prepared["data_sha256"]), (foundation, prepared["foundation_sha256"]),
                             (Path(manifest["background_cache"])/"index.json", prepared["backgrounds"]["index_sha256"])):
            if sha256(path) != digest:
                raise ValueError(f"Pinned input changed: {path}")
        if not protocol.get("regression_repair") or not a.video00004.is_file():
            raise ValueError("Source repair recipe or untouched Video00004 config missing")
        gpu = idle_gpu()
        command("preflight_offset", [sys.executable,"-u","-m","scripts.anti_uav.preflight_regression_repair",
            "--data",data,"--foundation",foundation,"--output",a.run_dir/"preflight",
            "--device",gpu,"--p3-nearest-offset"], gpu)
        acceptance = read(a.run_dir/"preflight/acceptance.json")
        if not acceptance["complete"] or acceptance["audit"]["balance_mix"] != protocol["regression_repair"]["options"]["balance_mix"]:
            raise ValueError("Real-batch P3 shift preflight failed")
        checkpoints, outcomes = {}, {}
        for name, option in (("control", []), ("p3_offset", ["--p3-nearest-offset"])):
            idle_gpu(only=gpu)
            arm = a.run_dir/name
            args = training_command(protocol, data, foundation, arm, gpu, prepared["training_videos"])
            command(f"train_{name}", args+["--foundation-only", *option], gpu)
            checkpoints[name] = arm/"foundation/p3/weights/last.pt"
            outcomes[name] = read(arm/"status.json")
        keys = ("epoch","batches_per_epoch","optimizer_updates")
        control, offset = [tuple(outcomes[name].get(k) for k in keys) for name in checkpoints]
        if control != offset or not all(control) or control[0] != protocol["epochs"]["foundation"]:
            raise ValueError("Training budgets differ; refuse to compare the final checkpoints")
        idle_gpu(only=gpu)
        command("evaluate_equal_fp", [sys.executable,"-u","-m","scripts.anti_uav.evaluate_p3_equal_fp",
            "--control",checkpoints["control"],"--offset",checkpoints["p3_offset"],"--device",gpu,
            "--video00004",a.video00004,"--video00009",data,"--output",a.run_dir/"comparison"], gpu)
        status("complete", matched_steps=dict(zip(keys,control)), training_videos=prepared["training_videos"],
               comparison=str(a.run_dir/"comparison/comparison.json"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
