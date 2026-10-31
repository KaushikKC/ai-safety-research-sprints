#!/usr/bin/env bash
# Full pipeline. Usage:  ./run_all.sh                      (Qwen2.5-1.5B-Instruct, local)
#                        MODEL=Qwen/Qwen2.5-7B-Instruct ./run_all.sh   (cloud GPU)
set -euo pipefail
cd "$(dirname "$0")"
TAG=$(basename "${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}")
mkdir -p logs
LOG="logs/run_${TAG}_$(date +%Y%m%d_%H%M).log"
echo "logging to $LOG"
{
  echo "== start $(date)"
  (cd src && python3 -u build_and_extract.py)
  echo "== extraction done $(date)"
  (cd src && python3 -u probes.py)
  echo "== done $(date)"
} 2>&1 | tee "$LOG"
