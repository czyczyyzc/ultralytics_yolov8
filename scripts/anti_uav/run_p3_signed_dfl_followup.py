"""Gate downstream training on a measured matched-step improvement on Video00009."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
from scripts.anti_uav.run_approved_gray_fixedquota_refresh import choose_shared_gpu, training_command

ROOT = Path(__file__).resolve().parents[2]
SIZES = ("long_lt4px", "long_4to8px")
BUDGETS = ("100", "200", "400")
PRIMARY = "200"


def read(path):
    return json.loads(Path(path).read_text())


def check_improvement(comparison):
    arms = comparison["arms"]
    baseline = arms["control"]["splits"]["Video00009"]["matched_fp"]
    candidate = arms["p3_signed_dfl"]["splits"]["Video00009"]["matched_fp"]
    evidence = {}
    for budget in BUDGETS:
        ref, new = baseline[budget], candidate[budget]
        if ref["frames"] != new["frames"] or ref["budget_fp"] != new["budget_fp"]:
            raise ValueError("Inconsistent Video00009 frame count or FP budgets")
        size_deltas = {}
        counts = [0, 0]
        total = 0
        for size in SIZES:
            old_size, new_size = ref["sizes"][size], new["sizes"][size]
            n = old_size["gt"]
            if n < 1 or n != new_size["gt"]:
                raise ValueError("Inconsistent Video00009 tiny GT counts")
            for i, field in enumerate(("tp50", "tp75")):
                counts[i] += new_size[field] - old_size[field]
            total += n
            size_deltas[size] = dict(
                gt=n, recall50_delta_pp=100*(new_size["tp50"]-old_size["tp50"])/n,
                recall75_delta_pp=100*(new_size["tp75"]-old_size["tp75"])/n,
                old_median_center_error_px=old_size["median_matched_center_error_input_px"],
                new_median_center_error_px=new_size["median_matched_center_error_input_px"])
        comparable = (max(ref["fp"],new["fp"]) <= ref["budget_fp"]
                      and abs(ref["fp_per_1000"]-new["fp_per_1000"]) <= 5.)
        evidence[budget] = dict(comparable=comparable, old_fp_per_1000=ref["fp_per_1000"],
            new_fp_per_1000=new["fp_per_1000"], pooled_recall50_delta_pp=100*counts[0]/total,
            pooled_recall75_delta_pp=100*counts[1]/total, sizes=size_deltas)

    row = evidence[PRIMARY]
    checks = dict(primary_fp_comparable=row["comparable"],
        pooled_recall50_gain=row["pooled_recall50_delta_pp"] >= 3.,
        no_size_recall50_regression=all(v["recall50_delta_pp"] >= -1. for v in row["sizes"].values()),
        no_pooled_recall75_regression=row["pooled_recall75_delta_pp"] >= -1.,
        no_size_recall75_regression=all(v["recall75_delta_pp"] >= -2. for v in row["sizes"].values()),
        stable_at_multiple_fp_budgets=sum(e["comparable"] and e["pooled_recall50_delta_pp"] >= 0.
                                          for e in evidence.values()) >= 2)
    return dict(passed=all(checks.values()), criteria=dict(primary_fp_per_1000=int(PRIMARY), max_fp_gap_per_1000=5.,
        min_pooled_recall50_gain_pp=3., max_size_recall50_drop_pp=1., max_pooled_recall75_drop_pp=1.,
        max_size_recall75_drop_pp=2., min_nonnegative_comparable_budgets=2), checks=checks, evidence=evidence)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--wait-for-foundation-complete", action="store_true",
                   help="Wait for the entire initial schedule before starting the accepted curriculum")
    a = p.parse_args()
    initial, run = a.initial_run.resolve(), a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run/"followup.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = run/"supervisor_status.json"

    def status(stage, **more):
        dump(output, dict(stage=stage, updated=time.time(), pid=os.getpid(), **more))

    try:
        initial_protocol = read(initial/"comparison_protocol.json")
        stopped = initial_protocol["stopped_control"]
        baseline = Path(initial_protocol["control"])
        candidate_receipt = initial/"candidate/audit/matched_epoch/snapshot.json"
        comparison = initial/f"comparison_epoch_{stopped['epoch']:04d}"/"comparison.json"
        while not comparison.is_file():
            initial_status = read(initial/"status.json")
            if initial_status["stage"] in {"failed", "matched_evaluation_failed_training_continues"}:
                raise ValueError(f"Initial experiment or matched evaluation failed: {initial_status.get('error')}")
            status("waiting_for_matched_evaluation", epoch=stopped["epoch"], initial_stage=initial_status["stage"],
                   criteria="200 FP/1000 primary, pooled tiny Recall@.50 +3 pp, no material Recall@.75 regression")
            time.sleep(60)
        candidate = read(candidate_receipt)
        result = read(comparison)
        if any(candidate[key] != stopped[key] for key in ("epoch", "optimizer_updates", "batches")):
            raise ValueError("Comparison steps do not match the control")
        for key, row in (("control", stopped), ("p3_signed_dfl", candidate)):
            evidence = result["arms"][key]
            if (evidence["weights"] != row["checkpoint"] or evidence["sha256"] != row["sha256"]
                    or sha256(row["checkpoint"]) != row["sha256"]):
                raise ValueError("Evaluation does not refer to the frozen matched checkpoint")
        data = Path(initial_protocol["data"])
        if sha256(data) != initial_protocol["data_sha256"] or result["splits"]["Video00009"]["sha256"] != sha256(data):
            raise ValueError("Video00009 data changed since the P3 comparison")
        report = check_improvement(result)
        report.update(candidate_checkpoint=candidate["checkpoint"], candidate_sha256=candidate["sha256"],
            comparison=str(comparison), comparison_sha256=sha256(comparison), training_videos=97,
            data_sha256=sha256(data), epoch=candidate["epoch"], optimizer_updates=candidate["optimizer_updates"])
        gate = run/"p3_acceptance.json"
        dump(gate, report)
        if not report["passed"]:
            status("p3_gate_not_met", report=str(gate), failed_checks=[k for k,v in report["checks"].items() if not v])
            return
        if a.wait_for_foundation_complete:
            while True:
                training = read(initial/"candidate/status.json")
                coordinator = read(initial/"status.json")
                if training["stage"] == "foundation_complete":
                    if (training["epoch"] != initial_protocol["foundation_epochs"]
                            or training["optimizer_updates"] != initial_protocol["optimizer_updates_scheduled"]):
                        raise ValueError("Foundation finished without the scheduled training budget")
                    break
                if training["stage"] == "failed" or coordinator["stage"] == "failed":
                    raise ValueError("Initial P3 did not complete after passing the matched gate")
                status("p3_accepted_waiting_for_foundation_complete", report=str(gate),
                       completed_epoch=training.get("epoch"), planned_epochs=initial_protocol["foundation_epochs"])
                time.sleep(60)
        while True:
            usage = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True)
            gpu = choose_shared_gpu(usage, a.gpus, False)
            if gpu is not None:
                break
            status("p3_accepted_waiting_for_idle_gpu", report=str(gate))
            time.sleep(60)
        protocol = read(baseline/"protocol.json")
        curriculum = run/"curriculum"
        command = training_command(protocol,data,protocol["foundation"],curriculum,gpu,97)
        command += ["--accepted-p3-gate", str(gate)]
        status("launching_accepted_teacher_then_student", gpu=gpu, report=str(gate), curriculum=str(curriculum))
        with (run/"curriculum.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status("curriculum_complete", report=str(gate), curriculum=str(curriculum))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
