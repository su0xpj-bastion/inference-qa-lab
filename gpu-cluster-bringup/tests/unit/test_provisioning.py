from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from bringup.pipeline import load_manifest

ROOT = Path(__file__).resolve().parents[2]
PROV = ROOT / "provision"


def _read(p: Path) -> str:
    with p.open("r", encoding="utf-8") as fh:
        return fh.read()


def test_cloud_init_renders_to_valid_yaml_with_the_hook():
    """The Terraform template must render to cloud-config that runs the acceptance hook."""
    tpl = _read(PROV / "aws-p5" / "cloud-init.yaml.tftpl")
    vars_ = {"cluster_name": "c", "node_type": "aws-p5-efa", "manifest_s3": "s3://b/m.json", "results_s3": "s3://b/r"}
    rendered = re.sub(r"\$\{(\w+)\}", lambda m: vars_[m.group(1)], tpl)
    assert rendered.startswith("#cloud-config")
    doc = yaml.safe_load(rendered)
    hook = next(f for f in doc["write_files"] if f["path"] == "/usr/local/sbin/bringup-hook")
    assert "python -m bringup node" in hook["content"] and "--stage node --stage domain" in hook["content"]
    assert "create-tags" in hook["content"]
    assert any("bringup-hook" in " ".join(c) for c in doc["runcmd"])


def test_cloud_init_node_type_exists_in_the_aws_manifest():
    tf = _read(PROV / "aws-p5" / "main.tf")
    node_type = re.search(r'node_type\s*=\s*"([^"]+)"', tf).group(1)
    m = load_manifest(ROOT / "manifests" / "aws-p5.json")
    assert node_type in m["node_types"]
    assert m["node_types"][node_type]["fabric"] == {"kind": "efa", "count": 32}
    assert "efa_cards = 32" in tf  # launch template and manifest agree on EFA count


def test_onprem_pins_match_the_golden_manifest():
    """Provisioning and acceptance must pin the same versions, or every node fails drift on day 1."""
    pins = yaml.safe_load(_read(PROV / "onprem" / "group_vars" / "all.yml"))
    g = load_manifest(ROOT / "manifests" / "golden.json")["node_types"]
    nv, amd = g["hgx-h100-ndr"]["software"], g["mi300x-ndr"]["software"]
    assert pins["nvidia_driver_version"].startswith(nv["driver"] + "-")
    assert pins["cuda_toolkit_package"].endswith(nv["runtime"].replace(".", "-"))
    assert pins["kernel_version"] == nv["kernel"] == amd["kernel"]
    assert pins["rocm_version"] == amd["runtime"] and pins["amdgpu_dkms_version"] == amd["driver"]


def test_onprem_inventory_node_types_are_known():
    known = set(load_manifest(ROOT / "manifests" / "golden.json")["node_types"])
    types = set(re.findall(r"node_type=(\S+)", _read(PROV / "onprem" / "inventory.ini")))
    assert types and types <= known


def test_acceptance_play_gates_on_report_exit_codes():
    plays = yaml.safe_load(_read(PROV / "onprem" / "acceptance.yml"))
    task = next(t for t in plays[1]["tasks"] if t["name"] == "Run the staged pipeline")
    assert task["failed_when"] == "bringup.rc not in [0, 3]"  # REJECTED (1) fails the play


def test_manifests_are_valid_json():
    for p in (ROOT / "manifests").glob("*.json"):
        with p.open("r", encoding="utf-8") as fh:
            assert json.load(fh)["schema"] == "golden-manifest/v1"
