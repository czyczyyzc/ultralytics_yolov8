"""Launch one fresh signed-DFL foundation after frozen expanded-data QA."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import yaml

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
from scripts.anti_uav.run_approved_gray_fixedquota_refresh import choose_shared_gpu, training_command

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def verify_not_returned(database, videos):
    """Use the frozen revisions, but do not launch with a subsequently returned clip."""
    later_edits = []
    with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as con:
        con.execute("BEGIN")
        for item in videos:
            current = con.execute("SELECT status,revision,is_published FROM tasks WHERE id=?",
                                  (item["task_id"],)).fetchone()
            if (current is None or current[2] != 1 or current[0] not in {"approved","submitted"}
                    or current[1] < item["revision"]
                    or (item["status"] == "approved" and current[:2] != ("approved",item["revision"]))):
                raise ValueError(f"Frozen task is no longer eligible: {item['task_id']}")
            if current[:2] != (item["status"],item["revision"]):
                rejected = con.execute("SELECT 1 FROM task_review_events WHERE task_id=? AND decision='reject' "
                                       "AND revision>=? LIMIT 1",(item["task_id"],item["revision"])).fetchone()
                if rejected:
                    raise ValueError(f"Frozen provisional revision was returned: {item['task_id']}")
                later_edits.append(dict(task_id=item["task_id"],frozen_status=item["status"],
                    frozen_revision=item["revision"],live_status=current[0],live_revision=current[1]))
    return later_edits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","preparation","run-dir","database"):
        p.add_argument(f"--{name}", required=True, type=lambda s: Path(s).resolve())
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    a = p.parse_args()
    a.run_dir.mkdir(parents=True,exist_ok=True)
    lock = (a.run_dir/"launch.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs = a.run_dir/"logs"
    logs.mkdir(exist_ok=True)

    def status(stage, **more):
        dump(a.run_dir/"status.json",dict(stage=stage,pid=os.getpid(),updated=time.time(),**more))

    def run(name, command):
        with (logs/f"{name}.log").open("x") as stream:
            subprocess.run(list(map(str,command)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

    try:
        while not (a.preparation/"preparation.json").is_file():
            prep = read(a.preparation/"refresh_status.json")
            if prep["stage"] == "failed":
                raise ValueError(f"Expanded data preparation failed: {prep['error']}")
            status("waiting_for_expanded_data",preparation_stage=prep["stage"])
            time.sleep(30)
        prepared = read(a.preparation/"preparation.json")
        data, weights = Path(prepared["data"]),Path(prepared["foundation"])
        recipe = read(a.baseline/"protocol.json")
        config = yaml.safe_load(data.read_text())
        dataset = read(data.parent/"manifest.json")
        base_data = Path(recipe["data"])
        baseline = read(base_data.parent/"manifest.json")
        frozen = read(data.parent/"annotation_snapshot.json")
        snapshot = frozen["videos"]
        hashes = {v["source_sha256"] for v in snapshot}
        holdout = {baseline["test_sha256"],baseline["validation_sha256"]}
        if (not prepared["complete"] or sha256(data) != prepared["data_sha256"]
                or sha256(weights) != prepared["foundation_sha256"] or sha256(weights) != recipe["foundation_sha256"]
                or hashes != set(prepared["requested_sha256"]) or len(hashes) != len(snapshot)
                or set(dataset["train_video_hashes"]) != set(baseline["train_video_hashes"])|hashes
                or holdout & set(dataset["train_video_hashes"])
                or config.get("visible_box_guard") != "affine_keep_subpixel_v2" or config.get("input_color") != "gray3"):
            raise ValueError("Prepared dataset differs from its frozen expansion contract")
        sampler = read(a.preparation/"audit/sampler_acceptance.json")
        qa_path = a.preparation/"audit/composition_acceptance.json"
        qa = read(qa_path)
        if (not sampler["complete"] or not sampler["public_coverage_complete"]
                or not sampler["all_native_positive_exposures_preserved"]
                or sha256(Path(config["online_replacement"]["cache"])/"index.json") != prepared["backgrounds"]["index_sha256"]
                or (prepared["visual_review_pending"] and
                    (qa["stage"] != "complete" or not qa["rendered"]
                     or len(qa["original_gt_previews"]) != len(snapshot)))):
            raise ValueError("Expanded sampler or replacement cache acceptance incomplete")
        if not recipe.get("regression_repair"):
            raise ValueError("Signed P3 requires the audited repair recipe")
        scope = dict(training_videos=prepared["training_videos"],new_tasks=Counter(v["status"] for v in snapshot),
            data=str(data),data_sha256=sha256(data),foundation_sha256=sha256(weights),
            batch_quotas=sampler["sampler"]["batch_quotas"],batches_per_epoch=sampler["sampler"]["batches"],
            epochs=recipe["epochs"]["foundation"],qa_sha256=sha256(qa_path),
            pending_positive_only=True,teacher_student_queued=False)
        dump(a.run_dir/"launch_contract.json",scope)
        while True:
            usage = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"],text=True)
            gpu = choose_shared_gpu(usage,a.gpus,False)
            if gpu is not None:
                break
            status("waiting_for_idle_gpu",training_videos=scope["training_videos"],usage=usage)
            time.sleep(60)
        verify_not_returned(a.database,snapshot)
        status("preflight_signed_p3",gpu=gpu,training_videos=scope["training_videos"])
        run("preflight",[sys.executable,"-u","-m","scripts.anti_uav.preflight_regression_repair",
            "--data",data,"--foundation",weights,"--output",a.run_dir/"preflight","--device",gpu,"--p3-signed-dfl"])
        acceptance = read(a.run_dir/"preflight/acceptance.json")
        if not acceptance["complete"] or not acceptance["audit"]["p3_signed_dfl"]["reserved"]:
            raise ValueError("Real-batch signed P3 preflight failed")
        dump(a.run_dir/"launch_annotation_check.json",dict(
            frozen_at=frozen["captured_at"], frozen_snapshot_sha256=sha256(data.parent/"annotation_snapshot.json"),
            later_provisional_revisions_not_mixed=verify_not_returned(a.database,snapshot)))
        command = training_command(recipe,data,weights,a.run_dir/"candidate",gpu,scope["training_videos"])
        status("training_p3_initial",gpu=gpu,training_videos=scope["training_videos"])
        run("training",command+["--foundation-only","--p3-signed-dfl"])
        result = read(a.run_dir/"candidate/status.json")
        if (result["stage"] != "foundation_complete" or result["epoch"] != scope["epochs"]
                or result["batches_per_epoch"] != scope["batches_per_epoch"]):
            raise ValueError("Expanded P3 did not complete its audited schedule")
        status("foundation_complete",training_videos=scope["training_videos"],best=result["foundation_best"])
    except BaseException as error:
        status("failed",error=repr(error))
        raise


if __name__ == "__main__":
    main()
