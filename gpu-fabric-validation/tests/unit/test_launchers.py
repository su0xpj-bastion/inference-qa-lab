from __future__ import annotations

import importlib.util
import re

import pytest
import yaml

from fabval import nccl_tests
from fabval.sim import BASE_ENV, ROCE_ENV
from tests.conftest import ENCODING, REPO

LAUNCH = REPO / "launch"


def load_yaml(rel):
    with (LAUNCH / rel).open("r", encoding=ENCODING) as fh:
        return [d for d in yaml.safe_load_all(fh) if d]


def text(rel):
    with (LAUNCH / rel).open("r", encoding=ENCODING) as fh:
        return fh.read()


def env_file(name):
    spec = importlib.util.spec_from_file_location("rayjob", LAUNCH / "ray" / "nccl_allreduce_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.load_env_file(LAUNCH / name)


def worker(job):
    return job["spec"]["mpiReplicaSpecs"]["Worker"]["template"]


@pytest.mark.parametrize("fabric", ["roce", "ib"])
def test_configmap_matches_env_file(fabric):
    """The k8s ConfigMap is the env file, key for key: one NCCL config for all three launchers."""
    _, env = env_file(f"nccl-{fabric}.env")
    cms = {d["metadata"]["name"]: d["data"] for d in load_yaml("k8s/nccl-env-configmap.yaml")}
    assert cms[f"nccl-env-{fabric}"] == env
    assert "NCCL_SOCKET_IFNAME" not in env


def test_simulator_uses_the_shipped_env():
    """Simulated runs record exactly the NCCL settings the launchers ship."""
    _, env = env_file("nccl-roce.env")
    sim = {k: v for k, v in {**BASE_ENV, **ROCE_ENV}.items() if k != "NCCL_SOCKET_IFNAME"}
    assert sim == env


@pytest.mark.parametrize("fabric", ["roce", "ib"])
def test_mpijob(fabric):
    """GPUs, RDMA resources, hugepages (req == limit), IPC_LOCK, /dev/shm, one worker per node,
    and every NCCL variable forwarded with mpirun -x (ssh-spawned ranks do not inherit env)."""
    [job] = load_yaml(f"k8s/mpijob-nccl-tests-{fabric}.yaml")
    _, env = env_file(f"nccl-{fabric}.env")
    assert job["kind"] == "MPIJob" and job["spec"]["slotsPerWorker"] == 8
    assert job["spec"]["mpiReplicaSpecs"]["Worker"]["replicas"] == 8
    tpl = worker(job)
    [c] = tpl["spec"]["containers"]
    res = c["resources"]
    assert res["requests"] == res["limits"]
    assert res["limits"]["nvidia.com/gpu"] == 8 and res["limits"]["hugepages-2Mi"] == "2Gi"
    rdma = [k for k in res["limits"] if k.startswith(("nvidia.com/rail", "rdma/"))]
    assert len(rdma) == (8 if fabric == "roce" else 1)
    assert "IPC_LOCK" in c["securityContext"]["capabilities"]["add"]
    vols = {v["name"]: v for v in tpl["spec"]["volumes"]}
    assert vols["dshm"]["emptyDir"]["medium"] == "Memory"
    assert vols["hugepages"]["emptyDir"]["medium"] == "HugePages-2Mi"
    anti = tpl["spec"]["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
    assert anti["topologyKey"] == "kubernetes.io/hostname"
    assert "kueue.x-k8s.io/podset-preferred-topology" in tpl["metadata"]["annotations"]
    if fabric == "roce":
        nets = tpl["metadata"]["annotations"]["k8s.v1.cni.cncf.io/networks"].split(",")
        assert nets == [f"roce-rail{r}" for r in range(8)]
    cmd = job["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]["containers"][0]["command"]
    forwarded = {cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-x"}
    assert set(env) | {"NCCL_SOCKET_IFNAME"} <= forwarded


def test_networks_one_per_rail():
    docs = load_yaml("k8s/roce-networks.yaml")
    assert [d["metadata"]["name"] for d in docs] == [f"roce-rail{r}" for r in range(8)]
    for r, d in enumerate(docs):
        assert d["metadata"]["annotations"]["k8s.v1.cni.cncf.io/resourceName"] == f"nvidia.com/rail{r}_vf"
        assert f'"range": "10.10{r}.0.0/16"' in d["spec"]["config"]


def test_raycluster_matches_mpijob_worker():
    """The Ray workers get the same GPUs, rails, hugepages, IPC_LOCK and ConfigMap as MPI workers."""
    [rc] = load_yaml("ray/raycluster.yaml")
    [job] = load_yaml("k8s/mpijob-nccl-tests-roce.yaml")
    rw = rc["spec"]["workerGroupSpecs"][0]["template"]
    mw = worker(job)
    assert rw["spec"]["containers"][0]["resources"] == mw["spec"]["containers"][0]["resources"]
    assert rw["spec"]["containers"][0]["envFrom"] == mw["spec"]["containers"][0]["envFrom"]
    assert rw["metadata"]["annotations"] == mw["metadata"]["annotations"]
    assert rc["spec"]["workerGroupSpecs"][0]["replicas"] == 8


def test_sbatch():
    """GPU/task shape, exclusive nodes, PMIx, pyxis container, and the shared env file."""
    s = text("slurm/nccl_tests.sbatch")
    for directive in ("--ntasks-per-node=8", "--gpus-per-node=8", "--exclusive", "--switches="):
        assert re.search(rf"^#SBATCH {re.escape(directive)}", s, re.M), directive
    assert "srun --mpi=pmix" in s and "--container-image=" in s
    assert 'source "${REPO_DIR}/launch/nccl-${FABRIC}.env"' in s and "set -a" in s
    assert "NCCL_SOCKET_IFNAME" in s


def test_pairwise_script_writes_analyzer_layout():
    s = text("slurm/pairwise.sh")
    assert "pairs/${a}__${b}__rail${r}.sendrecv_perf.txt" in s
    assert 'NCCL_IB_HCA="=mlx5_${r}"' in s and 'CUDA_VISIBLE_DEVICES="${r}"' in s


def test_ray_job_output_parses_as_nccl_tests():
    """Ray rows use nccl-tests layout and busbw math, so the same parser and gates apply."""
    mod, env = env_file("nccl-roce.env")
    devices = [(f"ray-{n}", d) for n in range(2) for d in range(8)]
    lines = [mod.format_header(8, 1 << 30, devices)]
    lines += [mod.format_row(s, 16, 30.0 + s / 190e3, 0) for s in mod.size_sweep(8, 1 << 30)]
    lines += ["# Out of bounds values : 0 OK", "# Avg bus bandwidth    : 1.0 ", "#"]
    r = nccl_tests.parse("\n".join(lines), "all_reduce")
    assert r.complete and r.nranks == 16 and len(r.rows) == 28
    assert r.consistency_problems() == []
    assert mod.runtime_env(LAUNCH / "nccl-roce.env")["env_vars"] == {**env, "NCCL_SOCKET_IFNAME": "eth0"}
