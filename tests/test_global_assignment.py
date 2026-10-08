import ctypes as ct
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def solver(tmp_path_factory):
    cxx=shutil.which("c++") or shutil.which("g++")
    if not cxx:
        pytest.skip("C++17 compiler required")
    src=ROOT/"scripts/anti_uav/dist_native"
    lib=tmp_path_factory.mktemp("global_assignment")/"solver.so"
    subprocess.run([cxx,"-std=c++17","-O2","-shared","-fPIC","-I",str(src),
        str(ROOT/"tests/native/global_assignment_api.cpp"),str(src/"third_party/lap/lapjv.cpp"),"-o",str(lib)],check=True)
    api=ct.CDLL(str(lib)).test_global_assignment
    api.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_void_p,ct.c_void_p,ct.c_void_p]
    def run(cost):
        data=np.ascontiguousarray(cost,dtype=np.float64);nr,nc=data.shape
        matches=np.empty(nr,np.int32);ambiguous=np.empty(nc,np.int32);counts=np.empty(8,np.uint64)
        assert api(data.ctypes.data,nr,nc,.7,.03,matches.ctypes.data,ambiguous.ctypes.data,counts.ctypes.data)==0
        return matches.tolist(),ambiguous.tolist(),counts.tolist()
    return run


def test_global_constraints_resolve_local_close_candidates(solver):
    matches,ambiguous,counts=solver([[.20,.21],[.90,.20]])
    assert matches==[0,1] and ambiguous==[0,0]
    assert counts[5]>0


def test_true_global_ambiguity_is_not_forced(solver):
    matches,ambiguous,_=solver([[.10,.30],[.20,.40]])
    assert matches==[-1,-1] and ambiguous==[1,1]


def test_dummy_permutations_are_not_identity_ambiguity(solver):
    matches,ambiguous,counts=solver([[.2,100],[100,.2]])
    assert matches==[0,1] and ambiguous==[0,0] and counts[4]==0


def test_unmatched_option_alone_does_not_manufacture_identity_ambiguity(solver):
    matches,ambiguous,_=solver([[.68,.69],[100,.2]])
    assert matches==[0,1] and ambiguous==[0,0]


def test_large_components_abstain_without_unbounded_solves(solver):
    matches,ambiguous,counts=solver(np.full((8,9),.2))
    assert matches==[-1]*8 and ambiguous==[1]*9
    assert counts[3]==0 and counts[4]==0 and counts[7]==9


def test_alternative_check_budget_is_bounded(solver):
    cost=np.full((9,18),100.)
    for i in range(9):cost[i,2*i:2*i+2]=[.2,.5]
    matches,ambiguous,counts=solver(cost)
    assert counts[4]==8
    assert matches[:8]==list(range(0,16,2)) and matches[8]==-1
    assert ambiguous[-2:]==[1,1] and counts[7]>0


def test_native_benchmark_builds_and_measures_only_association(tmp_path):
    import json
    from scripts.anti_uav.motion_native_runtime import DEFAULTS
    cxx=shutil.which("c++") or shutil.which("g++")
    if not cxx:
        pytest.skip("C++17 compiler required")
    src=ROOT/"scripts/anti_uav/dist_native";exe=tmp_path/"benchmark"
    subprocess.run([cxx,"-std=c++17","-O2","-ffp-contract=off",str(src/"motion_benchmark.cpp"),
        str(src/"motion_tracker.cpp"),str(src/"third_party/lap/lapjv.cpp"),"-o",str(exe)],check=True)
    config=tmp_path/"config.txt";config.write_text("14\n"+" ".join(str(v) for v in DEFAULTS.values())+"\n")
    frames=tmp_path/"frames.txt"
    frames.write_text("3\n"+"".join(f"{i/30} 1 1 0 0 0 1 0 1 100 100 104 104 .9\n" for i in range(3)))
    result=json.loads(subprocess.check_output([str(exe),str(config),str(frames),"2"],text=True))
    assert result["samples"]==6 and result["emitted_boxes_all_repeats"]==6
    assert result["confirmed_boxes_all_repeats"]==2
    assert "exposure" in result["scope"]
    assert 0<=result["milliseconds"]["p95"]<=result["milliseconds"]["max"]
