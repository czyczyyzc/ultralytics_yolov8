#!/usr/bin/env python3
"""Retrieve this running experiment's finished delivery and verify every file."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def unpack_verified(archive, destination):
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            target.relative_to(destination.resolve())
            if not (member.isdir() or member.isfile()):
                raise ValueError("Delivery archive may contain only regular files and directories")
        stream.extractall(destination)
    package = destination / archive.name.removesuffix(".tar.gz")
    for line in (package / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (package / relative).resolve()
        path.relative_to(package.resolve())
        if sha256(path) != digest:
            raise ValueError(f"Package member SHA256 mismatch: {relative}")
    return package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--workflow", required=True)
    p.add_argument("--expected-archive", required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--status-dir", type=Path, required=True)
    p.add_argument("--poll-seconds", type=int, default=120)
    a = p.parse_args()
    if a.poll_seconds < 30 or not PurePosixPath(a.workflow).is_absolute() or not PurePosixPath(a.expected_archive).is_absolute():
        raise ValueError("Require absolute remote paths and a non-aggressive poll interval")
    a.status_dir.mkdir(parents=True, exist_ok=True)
    a.destination.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 72*3600
    errors = 0
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", a.host,
                "cat " + shlex.quote(a.workflow)], text=True, capture_output=True, timeout=30, check=True)
            record = json.loads(result.stdout)
            dump(a.status_dir / "remote_workflow_status.json", record)
            errors = 0
            if record["stage"] == "failed":
                dump(a.status_dir / "fetch_status.json", dict(stage="remote_failed", remote=record))
                return
            if record["stage"] == "complete":
                if record["archive"] != a.expected_archive:
                    raise ValueError("Completed archive does not match the requested experiment")
                archive = a.destination / PurePosixPath(a.expected_archive).name
                dump(a.status_dir / "fetch_status.json", dict(stage="downloading", archive=str(archive)))
                subprocess.run(["rsync", "-az", "--partial", "--timeout=600", a.host + ":" + a.expected_archive,
                                str(a.destination) + "/"], check=True)
                if sha256(archive) != record["sha256"]:
                    raise ValueError("Downloaded archive SHA256 differs from completed workflow")
                package = unpack_verified(archive, a.destination)
                dump(a.status_dir / "fetch_status.json", dict(stage="complete", archive=str(archive),
                    package=str(package), handover=str(package / "DEPLOYMENT_HANDOVER_ZH.md"),
                    model=str(package / "models/detector_960x544_rk3576_w8a8.rknn")))
                return
            dump(a.status_dir / "fetch_status.json", dict(stage="waiting", remote_stage=record["stage"]))
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
            errors += 1
            dump(a.status_dir / "fetch_status.json", dict(stage="retrying", errors=errors, error=str(error)))
        time.sleep(min(a.poll_seconds * max(errors, 1), 600))
    dump(a.status_dir / "fetch_status.json", dict(stage="timeout", hours=72))


if __name__ == "__main__":
    main()
