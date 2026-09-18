from pathlib import Path

import pytest

from scripts.anti_uav.resume_online_gray_backgrounds import cpu_groups, ensure_quiescent, unique_physical_cores


def test_cpu_groups_are_disjoint():
    groups = cpu_groups("32-63", 16, range(128))
    assert len(groups) == 16
    assert groups[0] == [32, 33]
    assert groups[-1] == [62, 63]
    assert len({c for g in groups for c in g}) == 32


@pytest.mark.parametrize("cpus,workers", [("1,1", 1), ("1-3", 1), ("3-1", 1), ("128-129", 1)])
def test_invalid_affinity_is_rejected(cpus, workers):
    with pytest.raises(ValueError):
        cpu_groups(cpus, workers, range(128))


def test_reject_smt_siblings(tmp_path):
    for cpu in (0, 64):
        root = tmp_path/f"cpu{cpu}"/"topology"
        root.mkdir(parents=True)
        (root/"physical_package_id").write_text("0")
        (root/"core_id").write_text("0")
    with pytest.raises(ValueError, match="SMT"):
        unique_physical_cores([[0, 64]], tmp_path)


def test_active_old_writer_is_rejected(tmp_path):
    process = tmp_path/"12345"
    process.mkdir()
    root = Path("/data/cache")
    (process/"cmdline").write_bytes(b"python\0prepare_online_gray_replacement.py\0--output\0/data/cache\0")
    with pytest.raises(RuntimeError, match="still writing"):
        ensure_quiescent(root, tmp_path)
    ensure_quiescent(Path("/data/different-cache"), tmp_path)
