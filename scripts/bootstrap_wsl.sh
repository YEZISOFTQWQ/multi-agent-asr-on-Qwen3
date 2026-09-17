#!/usr/bin/env bash
# 在 WSL 中以幂等方式创建开发环境、目录和本地配置，并运行基础验证。
set -euo pipefail

# 这些变量允许开发者覆盖本机 Conda 位置和源/目标环境名称。
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"
SOURCE_ENV="${SOURCE_ENV:-qwen3-asr}"
TARGET_ENV="${TARGET_ENV:-multi-agent-asr}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="$MINIFORGE_ROOT/bin/conda"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda not found: $CONDA_BIN" >&2
  exit 1
fi

# 克隆已验证的 Qwen 环境可以复用 CUDA 依赖；已存在时保持原环境不变。
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$TARGET_ENV"; then
  "$CONDA_BIN" create --name "$TARGET_ENV" --clone "$SOURCE_ENV" --yes
fi

# editable 安装让源码修改立即生效，同时安装测试和格式检查工具。
TARGET_PYTHON="$MINIFORGE_ROOT/envs/$TARGET_ENV/bin/python"
"$TARGET_PYTHON" -m pip install -e "$PROJECT_ROOT[dev]"

# 运行数据放在仓库之外，避免把音频、数据库或实验输出提交到 Git。
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

# 只在首次初始化时复制模板，避免覆盖开发者的本地路径和设备配置。
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
fi

# 初始化持久化表并验证新环境；任一步失败都会因 set -e 立即终止。
"$TARGET_PYTHON" -m multi_agent_asr.cli init-db
"$TARGET_PYTHON" -m pytest "$PROJECT_ROOT/tests"
"$TARGET_PYTHON" -m ruff check "$PROJECT_ROOT/src" "$PROJECT_ROOT/tests"

echo "Environment ready: $TARGET_ENV"
echo "Activate with: source $MINIFORGE_ROOT/etc/profile.d/conda.sh && conda activate $TARGET_ENV"
