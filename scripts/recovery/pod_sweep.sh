#!/usr/bin/env bash
# One-shot H100 sweep runner. Usage: bash scripts/recovery/pod_sweep.sh <host> <port> [trials]
set -euo pipefail
host="${1:?pod host, e.g. 1.2.3.4}"
port="${2:?pod ssh port}"
trials="${3:-60}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
key="$HOME/.ssh/runpod_counterpoint"
ssh_opts=(-i "$key" -o StrictHostKeyChecking=accept-new -p "$port")
target="root@$host"

echo "== uploading dataset + script"
ssh "${ssh_opts[@]}" "$target" "mkdir -p /workspace/h1"
scp "${ssh_opts[@]}" "$root/results/recovery-training/"*.npz \
    "$root/results/recovery-training/manifest.json" "$target:/workspace/h1/"
scp "${ssh_opts[@]}" "$root/scripts/recovery/train_rnn.py" "$target:/workspace/h1/"

echo "== verifying GPU and launching sweep"
ssh "${ssh_opts[@]}" "$target" "cd /workspace/h1 && \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' && \
  nohup python train_rnn.py --trials $trials > sweep.log 2>&1 & sleep 1; tail -2 /workspace/h1/sweep.log"

echo "== follow: ssh -i $key -p $port $host tail -f /workspace/h1/sweep.log"
