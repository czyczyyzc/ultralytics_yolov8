#!/usr/bin/env python3
"""Append audited videos and reuse immutable, compatible background packs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.prepare_online_gray_replacement import atomic_json, build_plan, finalize, prepare_video
from scripts.anti_uav.synthesize_gray_drone_replacements import sha256


def incremental_snapshot(audit, source):
    allowed = set(source["train_video_hashes"])
    heldout = {source["test_sha256"], source["validation_sha256"]}
    if heldout != set(audit["heldout_sha256"]) or allowed & heldout:
        raise ValueError("Holdout identities changed")
    rows = [r for r in audit["new_rows"] if r["sha256"] not in allowed]
    hashes = [r["sha256"] for r in rows]
    if not rows or len(set(hashes)) != len(hashes) or heldout & set(hashes):
        raise ValueError("Empty, duplicate or held-out expansion")
    if any(not r.get("video_and_annotation_hashes_valid") or not r.get("coco_yolo_geometry_valid")
           or r.get("positive_frames", 0) <= 0 for r in rows):
        raise ValueError("Expansion requires fully audited positive videos")
    return dict(audit, new_rows=rows, new_videos=len(rows),
                expanded_training_videos=len(allowed)+len(rows),
                new_positive_frames=sum(r["positive_frames"] for r in rows),
                new_negative_frames=sum(r["negative_frames"] for r in rows))


def compatible_video(old, new):
    if old["record"]["sha256"] != new["record"]["sha256"] or old["annotation_sha256"] != new["annotation_sha256"]:
        raise ValueError("Cached video/annotation identity changed")
    keys = ("frame", "box", "width", "height")
    if [{k: r[k] for k in keys} for r in old["positive"]] != [
            {k: r[k] for k in keys} for r in new["positive"]]:
        raise ValueError("Reviewed positive geometry changed")
    for before, after in zip(old["positive"], new["positive"]):
        # Unextracted historical frames live in the old cache, which remains in use.
        if after.get("image") and (after["image"] != before.get("image") or after["label"] != before.get("label")):
            raise ValueError("Source frame mapping changed; cannot reuse cache")


def reuse_packs(previous, root, plan):
    summary = json.loads((previous/"summary.json").read_text())
    if summary["stage"] != "complete" or summary["is_smoke_subset"]:
        raise ValueError("Can only reuse a completed full cache")
    if sha256(previous/"index.json") != summary["index_sha256"]:
        raise ValueError("Previous index changed")
    old = json.loads((previous/"plan.json").read_text())
    for key in ("code_sha256", "catalog_sha256", "asset_ids", "heldout_sha256"):
        if old[key] != plan[key]:
            raise ValueError(f"Incompatible cache algorithm/assets/holdout: {key}")
    mapping = {v["record"]["sha256"]: v for v in old["videos"]}
    allowed = {v["record"]["sha256"] for v in plan["videos"]}
    if not set(mapping) <= allowed:
        raise ValueError("Previous cache contains videos outside the new training split")
    reused, pending = [], []
    for item in plan["videos"]:
        digest = item["record"]["sha256"]
        if digest not in mapping:
            pending.append(item)
            continue
        compatible_video(mapping[digest], item)
        pack = previous/f"{digest}.sqlite"
        with sqlite3.connect(f"file:{pack}?mode=ro", uri=True) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError(f"Corrupt cache pack: {pack}")
            expected = {r["frame"]: r for r in item["positive"]}
            seen = set()
            for frame, image, label, box, state in db.execute("SELECT frame,image,label,box,state FROM backgrounds"):
                if frame not in expected or frame in seen or json.loads(box) != expected[frame]["box"]:
                    raise ValueError("Cache rows do not match reviewed frames")
                if state not in ("ready", "original_only") or not Path(image).is_file() or not Path(label).is_file():
                    raise ValueError("Missing original or invalid cache state")
                if expected[frame].get("image") and (image != expected[frame]["image"] or label != expected[frame]["label"]):
                    raise ValueError("Cache image/label mapping differs")
                seen.add(frame)
            if seen != set(expected):
                raise ValueError("Incomplete cached video")
        destination = root/pack.name
        if destination.is_symlink():
            if destination.resolve() != pack.resolve():
                raise ValueError("Existing cache link points elsewhere")
        elif destination.exists():
            raise FileExistsError(destination)
        else:
            destination.symlink_to(pack.resolve())
        reused.append(digest)
    return reused, pending


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--previous-cache", type=Path, required=True)
    p.add_argument("--native-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        raise ValueError("Use 1..8 independent video workers")
    for key in ("source", "audit", "previous_cache", "native_output", "output"):
        setattr(a, key, getattr(a, key).resolve())
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/"coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = dict(source=str(a.source), audit=str(a.audit), audit_sha256=sha256(a.audit),
                    source_manifest_sha256=sha256(a.source/"manifest.json"),
                    previous_cache=str(a.previous_cache), native_output=str(a.native_output),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    path = a.output/"extension_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Cannot resume changed extension inputs/code")
    atomic_json(path, protocol)
    state = dict(stage="preparing_native", pid=os.getpid(), training_started=False)
    def status(stage, **extra):
        state.update(stage=stage, **extra)
        atomic_json(a.output/"extension_status.json", state)
        print(json.dumps(state), flush=True)
    try:
        source = json.loads((a.source/"manifest.json").read_text())
        audit = json.loads(a.audit.read_text())
        original = Path(audit["source_dataset"])/"manifest.json"
        if sha256(original) != audit["source_manifest_sha256"]:
            raise ValueError("Audited baseline changed")
        if not set(json.loads(original.read_text())["train_video_hashes"]) <= set(source["train_video_hashes"]):
            raise ValueError("Incremental source does not contain the audited baseline")
        snapshot = incremental_snapshot(audit, source)
        atomic_json(a.output/"incremental_snapshot.json", snapshot)
        status("preparing_native", new_videos=snapshot["new_videos"], total_videos=snapshot["expanded_training_videos"])
        if not a.native_output.exists():
            subprocess.run([sys.executable, str(ROOT/"scripts/anti_uav/append_approved_gray_native.py"),
                "--source", str(a.source), "--snapshot", str(a.output/"incremental_snapshot.json"),
                "--output", str(a.native_output), "--video-root", "/mnt/andrew/video-labeler/videos",
                "--old-root", "/mnt/andrew/anti_uav_model_refinement/data/seven_old_videos"], check=True)
        else:
            meta = json.loads((a.native_output/"manifest.json").read_text())
            if meta["snapshot_sha256"] != sha256(a.output/"incremental_snapshot.json") or meta["source_dataset"] != str(a.source):
                raise ValueError("Existing native output is incomplete or belongs to another run")
        previous_plan = json.loads((a.previous_cache/"plan.json").read_text())
        plan = build_plan(a.native_output, Path(previous_plan["catalog"]))
        if (a.output/"plan.json").exists() and json.loads((a.output/"plan.json").read_text()) != plan:
            raise ValueError("Frozen extension plan changed")
        atomic_json(a.output/"plan.json", plan)
        status("checking_reusable_packs")
        reused, pending = reuse_packs(a.previous_cache, a.output, plan)
        status("preparing_backgrounds", reused_videos=len(reused), pending_videos=len(pending), completed_videos=[])
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(prepare_video, plan, v, str(a.output)) for v in pending]
            for future in as_completed(futures):
                state["completed_videos"].append(future.result())
                status("preparing_backgrounds")
        summary = finalize(plan, plan["videos"], a.output, False)
        status("complete", summary={k: v for k, v in summary.items() if k != "rejection_reasons"})
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
