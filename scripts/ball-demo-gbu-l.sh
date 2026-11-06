#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${TACSE3_PYTHON:-/home/gbu-l/anaconda3/envs/lerobot101/bin/python}
mkdir -p "$repo_dir/runs"
session_dir=$(mktemp -d "$repo_dir/runs/ball-XXXXXXXX")
cd "$session_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
printf 'Session: %s\n' "$session_dir"
"$python_bin" -B -u -m tacse3.ball_demo --sensor-serial M2505150299 M2505150016 --output "$session_dir/data" "$@"
