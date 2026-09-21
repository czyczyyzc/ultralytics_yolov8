#!/usr/bin/env python3
"""Create an isolated expanded-asset view without changing training exposures."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import yaml

from scripts.anti_uav.build_gray_replacement_batch import load_assets
from scripts.anti_uav.synthesize_gray_drone_replacements import safe_path, sha256


def select_records(original, candidates, original_ids):
    original_ids = set(original_ids)
    rows = [("original", r) for r in original["records"] if r["id"] in original_ids]
    if len(rows) != len(original_ids):
        raise ValueError("Original enabled IDs are missing or duplicated")
    for row in candidates["records"]:
        if row.get("status") != "visually_screened_compositing_candidate" or not row.get("cutout"):
            raise ValueError("Use the screened candidate catalog, not rejected/unreviewed raw outputs")
        rows.append(("candidate", row))
    ids = [r["id"] for _, r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping asset IDs")
    return sorted(rows, key=lambda pair: int(pair[1]["id"]))


def assert_asset_only_configs(before, after):
    a, b = deepcopy(before), deepcopy(after)
    for value in (a, b):
        value.pop("path", None)
        for key in ("train", "val"):
            value[key] = sha256(Path(value[key]))
        sampling = value.get("label_sampling")
        if sampling:
            sampling["negative_pool"] = sha256(Path(sampling["negative_pool"]))
        replacement = value.get("online_replacement")
        if not replacement:
            raise ValueError("Both arms must use online replacement")
        replacement.pop("cache")
    if a != b:
        raise ValueError("Asset-only experiment changed data, sampling or augmentation policy")


def prepare(source, candidates, output):
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((source/"summary.json").read_text())
    if summary["stage"] != "complete" or summary["is_smoke_subset"]:
        raise ValueError("Source cache is not a completed full cache")
    if sha256(source/"index.json") != summary["index_sha256"]:
        raise ValueError("Source cache index changed")
    index = json.loads((source/"index.json").read_text())
    original_catalog = Path(index["catalog"])
    if sha256(original_catalog) != index["catalog_sha256"]:
        raise ValueError("Original catalog changed")
    if [r["id"] for r, _ in load_assets(original_catalog, {"24", "25"})] != index["asset_ids"]:
        raise ValueError("Original active assets changed")
    if any(v["video_sha256"] in index["heldout_sha256"] for v in index["images"].values()):
        raise ValueError("Holdout found in source cache")
    inputs = {"original": original_catalog, "candidate": candidates}
    hashes = {k: sha256(v) for k, v in inputs.items()}
    records = select_records(json.loads(original_catalog.read_text()), json.loads(candidates.read_text()), index["asset_ids"])
    output.mkdir(parents=True)
    catalog_root = output/"assets"
    (catalog_root/"cutouts").mkdir(parents=True)
    merged = []
    for kind, row in records:
        path = safe_path(inputs[kind].parent, row["cutout"])
        if sha256(path) != row["cutout_sha256"]:
            raise ValueError(f"Cutout checksum mismatch: {path}")
        with Image.open(path) as im:
            if im.mode != "RGBA":
                raise ValueError(f"Missing alpha: {path}")
            alpha = np.asarray(im)[:, :, 3]
            if not np.any(alpha > 0) or not np.any(alpha < 255):
                raise ValueError(f"Empty or opaque cutout: {path}")
        rel = f"cutouts/{row['id']}.png"
        shutil.copy2(path, catalog_root/rel)
        merged.append(dict(id=row["id"], model=row["model"], company=row.get("company"),
            category=row.get("category"), cutout=rel, cutout_sha256=sha256(catalog_root/rel),
            source_catalog=str(inputs[kind]), source_catalog_sha256=hashes[kind],
            source_cutout=str(path), source_status=row.get("status"), experimental_candidate=(kind == "candidate")))
    catalog = catalog_root/"catalog.json"
    catalog.write_text(json.dumps(dict(records=merged, enabled_for_this_experiment=True,
        production_qualification=False, original_asset_count=len(index["asset_ids"]),
        candidate_asset_count=len(records)-len(index["asset_ids"])), indent=2)+"\n")
    enabled = [r["id"] for r, _ in load_assets(catalog, {"24", "25"})]
    if enabled != [r["id"] for r in merged]:
        raise ValueError("Expanded catalog contains a disabled asset")
    for pack in {r["pack"] for r in index["images"].values()}:
        if Path(pack).name != pack or not (source/pack).is_file():
            raise ValueError("Invalid source cache pack")
        (output/pack).symlink_to((source/pack).resolve())
    index.update(catalog=str(catalog), catalog_sha256=sha256(catalog), asset_ids=enabled)
    (output/"index.json").write_text(json.dumps(index, indent=2)+"\n")
    config = yaml.safe_load((source/"train_online_gray_monitor.yaml").read_text())
    expanded = deepcopy(config)
    expanded["online_replacement"]["cache"] = str(output)
    assert_asset_only_configs(config, expanded)
    (output/"train_online_gray_monitor.yaml").write_text(yaml.safe_dump(expanded, sort_keys=False))
    for name in ("train_hardneg.txt", "val_monitor.txt", "new_negative_pool.txt"):
        (output/name).symlink_to((source/name).resolve())
    manifest = json.loads((source/"manifest.json").read_text())
    manifest.update(schema="gray_asset_ablation_view.v1", source_dataset=str(source),
                    background_cache=str(output), online_replacement=expanded["online_replacement"], training_started=False)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    summary.update(asset_count=len(enabled), index_sha256=sha256(output/"index.json"),
                   training_started=False, manual_visual_review_pending=True, source_cache=str(source),
                   original_asset_count=len(index["asset_ids"])-sum(r["experimental_candidate"] for r in merged),
                   candidate_asset_count=sum(r["experimental_candidate"] for r in merged))
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    audit = dict(source_cache=str(source), source_index_sha256=sha256(source/"index.json"),
                 source_catalog_hashes=hashes, active_asset_ids=enabled,
                 training_entries_unchanged=True, validation_entries_unchanged=True,
                 negative_pool_unchanged=True, original_background_packs_reused=True,
                 replacement_probability=expanded["online_replacement"]["replacement_probability"],
                 source_files_modified=False, production_qualification=False)
    if any(sha256(inputs[k]) != h for k, h in hashes.items()):
        raise ValueError("Catalog changed during preparation")
    (output/"asset_ablation.json").write_text(json.dumps(audit, indent=2)+"\n")
    (output/"extension_status.json").write_text(json.dumps(dict(stage="complete", pid=os.getpid(),
        training_started=False, asset_count=len(enabled), source_cache=str(source)), indent=2)+"\n")
    return dict(asset_count=len(enabled), candidate_asset_count=summary["candidate_asset_count"],
                ready_backgrounds=summary["ready_backgrounds"], epoch_total=summary["epoch_total"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source.resolve(), a.candidates.resolve(), a.output.resolve())))


if __name__ == "__main__":
    main()
