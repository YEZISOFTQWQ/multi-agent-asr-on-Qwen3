#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"
SOURCE_ENV="${SOURCE_ENV:-qwen3-asr}"
TARGET_ENV="${TARGET_ENV:-multi-agent-asr}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="$MINIFORGE_ROOT/bin/conda"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda not found: $CONDA_BIN" >&2
  exit 1
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$TARGET_ENV"; then
  "$CONDA_BIN" create --name "$TARGET_ENV" --clone "$SOURCE_ENV" --yes
fi

TARGET_PYTHON="$MINIFORGE_ROOT/envs/$TARGET_ENV/bin/python"
"$TARGET_PYTHON" -m pip install -e "$PROJECT_ROOT[dev]"

mkdir -p \
  "$HOME/data/multi-agent-asr/raw" \
  "$HOME/data/multi-agent-asr/processed" \
  "$HOME/data/multi-agent-asr/annotations" \
  "$HOME/data/multi-agent-asr/manifests" \
  "$HOME/data/multi-agent-asr/state" \
  "$HOME/runs/multi-agent-asr/logs" \
  "$HOME/runs/multi-agent-asr/outputs" \
  "$HOME/runs/multi-agent-asr/checkpoints" \
  "$HOME/runs/multi-agent-asr/evaluations"

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
fi

"$TARGET_PYTHON" -m multi_agent_asr.cli init-db
"$TARGET_PYTHON" -m pytest "$PROJECT_ROOT/tests"
"$TARGET_PYTHON" -m ruff check "$PROJECT_ROOT/src" "$PROJECT_ROOT/tests"

echo "Environment ready: $TARGET_ENV"
echo "Activate with: source $MINIFORGE_ROOT/etc/profile.d/conda.sh && conda activate $TARGET_ENV"
