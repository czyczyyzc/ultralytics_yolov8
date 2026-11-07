#!/usr/bin/env python3
"""Read-only replay of the public-aircraft experiment's actual sampling order."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.audit_labeler_expansion_regression import (
    AuditDataset, label_equal, size_category, strip_locations,
)
from scripts.anti_uav.label_pool_sampling import TinyAwareSampler, EvenExposureTinyAwareSampler
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def source_kind(path, public):
    if path in public:
        return "public/" + public[path]["dataset"]
    if "/anti_uav300_yolo/" in path:
        return "native_antiuav_rgb"
    return "native_gray"


def compact_counts(values):
    counts = dict(Counter(values))
    return dict(counts=counts, fractions={k: v / max(sum(counts.values()), 1) for k, v in counts.items()})


def trace(labels, config, public, epoch, even=False):
    cls = EvenExposureTinyAwareSampler if even else TinyAwareSampler
    sampler = cls(AuditDataset(labels), config["label_sampling"], config["tiny_sampling"], 64, 20261015)
    sampler.set_epoch(epoch)
    order = list(sampler)
    domains = [source_kind(row["im_file"], public) for row in labels]
    categories = [size_category(row) for row in labels]

    def describe(indices):
        return dict(slots=len(indices), domains=compact_counts(domains[i] for i in indices),
                    sizes=compact_counts(categories[i] for i in indices),
                    positive_domains=compact_counts(domains[i] for i in indices if categories[i] != "negative"),
                    tiny_domains=compact_counts(domains[i] for i in indices if categories[i] == "4to8"),
                    public_fraction=sum(domains[i].startswith("public/") for i in indices) / max(len(indices), 1),
                    gray_fraction=sum(domains[i] == "native_gray" for i in indices) / max(len(indices), 1))

    groups = defaultdict(Counter)
    for i in order:
        groups[domains[i]][sampler.video_keys[i]] += 1
    longest, streak = 0, 0
    absent = 0
    batches = 0
    for start in range(0, len(order), 64):
        batch = order[start:start + 64]
        if len(batch) < 64:
            continue
        batches += 1
        no_gray = all(domains[i] != "native_gray" for i in batch)
        absent += no_gray
        streak = streak + 1 if no_gray else 0
        longest = max(longest, streak)
    return dict(epoch_zero_based=epoch, counterfactual_even_order=even, whole=describe(order),
                deciles=[describe(order[len(order)*j//10:len(order)*(j+1)//10]) for j in range(10)],
                last_1pct=describe(order[len(order)*99//100:]),
                no_gray_batches=absent, full_batches=batches, max_consecutive_no_gray_batches=longest,
                source_groups={domain: dict(groups=len(group), largest=group.most_common(3))
                               for domain, group in groups.items()},
                index_sha256=__import__("hashlib").sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest())


def box_distribution(labels, public):
    output = defaultdict(lambda: dict(frames=0, positive_frames=0, gt=0, bins=Counter(), edges=[]))
    for row in labels:
        domain = source_kind(row["im_file"], public)
        out = output[domain]
        out["frames"] += 1
        if not len(row["cls"]):
            continue
        out["positive_frames"] += 1
        h, w = row["shape"]
        boxes = np.asarray(row["bboxes"])
        edges = np.maximum(boxes[:, 2]*w, boxes[:, 3]*h)*min(544/h, 960/w)
        out["gt"] += len(edges)
        out["edges"].extend(edges.tolist())
        for edge in edges:
            key = "lt4" if edge < 4 else "4to8" if edge <= 8 else "8to16" if edge <= 16 else "16to64" if edge <= 64 else "gt64"
            out["bins"][key] += 1
    for out in output.values():
        edges = out.pop("edges")
        out["bins"] = dict(out["bins"])
        out["edge_p10_p50_p90"] = np.percentile(edges, [10, 50, 90]).tolist() if edges else []
    return dict(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-run", required=True, type=Path)
    p.add_argument("--new-run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, nargs="+", default=[0, 29])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    public = {}
    new_protocol = json.loads((args.new_run/"protocol.json").read_text())
    data_dir = Path(new_protocol["data"]).parent
    for row in json.loads((data_dir/"public_accepted.json").read_text()):
        public[row["image"]] = row
    report = dict(training_unmodified=True, public_images=len(public), runs={}, integrity={}, recipe_differences={})
    configs, labels, protocols = {}, {}, {}
    for name, run in (("native69", args.old_run), ("all_aircraft", args.new_run)):
        protocols[name] = json.loads((run/"protocol.json").read_text())
        configs[name] = yaml.safe_load((run/"configs/foundation.yaml").read_text())
        listing = Path(configs[name]["train"])
        pin = configs[name]["trusted_label_caches"][str(listing)]
        assert sha256(listing) == pin["source_sha256"]
        assert sha256(listing.with_suffix(".cache")) == pin["cache_sha256"]
        labels[name] = np.load(listing.with_suffix(".cache"), allow_pickle=True).item()["labels"]
        assert Counter(row["im_file"] for row in labels[name]) == Counter(listing.read_text().splitlines())
        report["runs"][name] = dict(box_distribution=box_distribution(labels[name], public), traces=[])
        for epoch in args.epochs:
            for even in (False, True):
                t = trace(labels[name], configs[name], public, epoch, even)
                report["runs"][name]["traces"].append(t)
                dump(args.output/"partial.json", report)
                print(json.dumps(dict(run=name, epoch=epoch, even=even, whole=t["whole"],
                                      last_decile=t["deciles"][-1], last_1pct=t["last_1pct"],
                                      no_gray_batches=t["no_gray_batches"], max_no_gray=t["max_consecutive_no_gray_batches"])), flush=True)
    a, b = ({row["im_file"]: row for row in labels[name]} for name in ("native69", "all_aircraft"))
    old_list = Path(configs["native69"]["train"]).read_text().splitlines()
    new_list = Path(configs["all_aircraft"]["train"]).read_text().splitlines()
    report["integrity"] = dict(missing_old_paths=len(a.keys()-b.keys()),
        changed_old_labels=sum(not label_equal(a[path], b[path]) for path in a.keys() & b.keys()),
        repeated_old_prefix_exact=new_list[:len(old_list)] == old_list,
        same_val_bytes=Path(configs["native69"]["val"]).read_bytes() == Path(configs["all_aircraft"]["val"]).read_bytes(),
        same_initialization=protocols["native69"]["foundation_sha256"] == protocols["all_aircraft"]["foundation_sha256"])
    for stage in ("foundation/p3", "teacher/p2p3", "distill_fp32/p3_student"):
        a, b = [strip_locations(yaml.safe_load((run/stage/"args.yaml").read_text()))
                for run in (args.old_run, args.new_run)]
        report["recipe_differences"][stage] = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    dump(args.output/"audit.json", report)
    print("AUDIT_COMPLETE", str(args.output/"audit.json"), flush=True)


if __name__ == "__main__":
    main()
