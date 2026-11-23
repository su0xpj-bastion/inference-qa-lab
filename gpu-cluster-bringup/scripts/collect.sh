#!/usr/bin/env bash
# Capture acceptance evidence on a real GPU node, in the layout bringup/collect.py reads.
#
#   sudo scripts/collect.sh <out_dir> <rack> <node_type> [burnin_seconds]
#
# Runs as root (lspci -vv hides LnkCap/LnkSta otherwise). Every tool's output is
# saved verbatim; a tool that fails leaves its file absent, and the node gate
# then fails closed on "missing evidence". This script has not been run on
# hardware from this repo's CI; it is the contract the parsers are written to.
set -euo pipefail

out=${1:?out_dir}
rack=${2:?rack}
node_type=${3:?node_type}
burnin_s=${4:-14400}
mkdir -p "$out"
host=$(hostname -s)

printf '{"hostname": "%s", "rack": "%s", "node_type": "%s"}\n' "$host" "$rack" "$node_type" > "$out/meta.json"
uname -r > "$out/uname-r.txt"
lspci -vv > "$out/lspci-vv.txt" 2>/dev/null || rm -f "$out/lspci-vv.txt"

# RDMA devices: ibstat for IB/RoCE, fi_info for AWS EFA. Map device -> PCI address.
if command -v ibstat >/dev/null; then ibstat > "$out/ibstat.txt" || rm -f "$out/ibstat.txt"; fi
if command -v fi_info >/dev/null && fi_info -p efa >/dev/null 2>&1; then
  fi_info -p efa > "$out/fi_info-efa.txt"
fi
: > "$out/ib-devices.txt"
for d in /sys/class/infiniband/*; do
  [ -e "$d" ] || continue
  echo "$(basename "$d") $(basename "$(readlink -f "$d/device")")" >> "$out/ib-devices.txt"
done

if command -v nvidia-smi >/dev/null; then
  nvidia-smi -q -x > "$out/nvidia-smi-q.xml" || rm -f "$out/nvidia-smi-q.xml"
  nvidia-smi topo -m > "$out/nvidia-smi-topo.txt" || rm -f "$out/nvidia-smi-topo.txt"
  # Level 3 is ~30 min on 8 GPUs; it is the burn-in stage's evidence, not a smoke test.
  dcgmi diag -r 3 > "$out/dcgmi-diag-r3.txt" 2>&1 || true
  burn_cmd=(gpu_burn -d "$burnin_s")
elif command -v rocm-smi >/dev/null; then
  rocm-smi --showproductname --showbus --showvbios --showserial --showtemp --showpower \
           --showmaxpower --showtoponuma --showrasinfo all --showdriverversion --json \
           > "$out/rocm-smi.json" || rm -f "$out/rocm-smi.json"
  rocm-smi --showtopotype --json > "$out/rocm-smi-topo.json" || rm -f "$out/rocm-smi-topo.json"
  cat /opt/rocm/.info/version > "$out/rocm-version.txt" 2>/dev/null || true
  burn_cmd=(rvs -c /opt/rocm/share/rocm-validation-suite/conf/gst_single.conf)
fi

# Burn-in: the soak tool's own log is kept; burnin.json is produced by the site's
# summariser (per-GPU max temp, throttle seconds, GEMM TFLOPS, errors).
dmesg_before=$(dmesg | wc -l)
if [ -n "${burn_cmd[*]:-}" ] && command -v "${burn_cmd[0]}" >/dev/null; then
  "${burn_cmd[@]}" > "$out/burnin.log" 2>&1 || true
fi
dmesg | tail -n +"$((dmesg_before + 1))" | grep -E 'NVRM: Xid|amdgpu.*(ras|RAS)' > "$out/dmesg-xid.txt" || true

# Single-node all-reduce over NVLink/XGMI (nccl-tests / rccl-tests binary of the same name).
if command -v all_reduce_perf >/dev/null; then
  all_reduce_perf -b 8M -e 8G -f 4 -g 8 -c 1 > "$out/all_reduce_perf.txt" 2>&1 || true
fi
echo "captures written to $out"
