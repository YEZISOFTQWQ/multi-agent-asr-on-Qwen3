# Multi-Agent ASR

这是一个建立在 Qwen3-ASR 之上的多智能体语音识别开发仓库。Qwen3-ASR 负责转写；本项目负责音频检查、说话人线索、场景线索、历史记忆、上下文构建、结果校验和 API 编排。

## 当前能力

- 懒加载 Qwen3-ASR，启动 API 时不会立即占用 GPU。
- 使用 SQLite 保存说话人画像和最近转写。
- 把说话人、场景、常用术语、历史纠错和最近对话整理成 `context`。
- 提供 FastAPI 健康检查、画像管理和转写接口。
- 说话人与场景智能体已经定义稳定接口，当前基础实现接受上游提示，后续可接入 pyannote、SpeechBrain、AST 或 PANNs。
- 校验智能体当前执行基础完整性检查，后续可以接入 ForcedAligner 和二次解码。

## 仓库边界

```text
./qwen3-asr       Qwen3-ASR 底层源码
./multi-agent-asr 本项目源码
./data/multi-agent-asr 运行数据与 SQLite
./runs/multi-agent-asr 实验输出与日志
```

模型权重由 Hugging Face 缓存管理，不提交到本仓库。

## 环境

首次初始化推荐从已经验证过的 `qwen3-asr` 环境克隆：

```bash
cd ./multi-agent-asr
bash scripts/bootstrap_wsl.sh
```

以后进入环境：

```bash
source ./miniforge3/etc/profile.d/conda.sh
conda activate multi-agent-asr
```

配置文件：

```bash
cp .env.example .env
```

默认模型为 `Qwen/Qwen3-ASR-1.7B`。如需时间戳，在 `.env` 中设置：

```text
MASR_FORCED_ALIGNER_MODEL_PATH=Qwen/Qwen3-ForcedAligner-0.6B
```

## 初始化与验证

```bash
multi-agent-asr init-db
python -m pytest
python -m ruff check src tests
```

## 启动 API

```bash
multi-agent-asr serve
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

写入说话人画像：

```bash
curl -X PUT http://127.0.0.1:8000/v1/profiles/speaker_001 \
  -H 'Content-Type: application/json' \
  -d '{
    "speaker_id": "speaker_001",
    "display_name": "张三",
    "accent": "四川口音",
    "accent_confidence": 0.81,
    "frequent_terms": ["Qwen3-ASR", "ForcedAligner"],
    "corrections": {"福斯阿莱纳": "ForcedAligner"},
    "environments": ["汽车驾驶舱"]
  }'
```

转写本地音频：

```bash
curl -X POST http://127.0.0.1:8000/v1/transcriptions \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_path": "./data/multi-agent-asr/raw/example.wav",
    "session_id": "demo-session",
    "speaker_hint": "speaker_001",
    "scene_hint": "汽车驾驶舱",
    "language": "Chinese"
  }'
```

## 命令行转写

```bash
multi-agent-asr transcribe \
  ./data/multi-agent-asr/raw/example.wav \
  --session-id demo-session \
  --speaker speaker_001 \
  --language Chinese
```

## 开发入口

- `src/multi_agent_asr/agents/orchestrator.py`：整体流程。
- `src/multi_agent_asr/services/qwen_service.py`：Qwen3-ASR 模型封装。
- `src/multi_agent_asr/memory/repository.py`：SQLite 持久化。
- `src/multi_agent_asr/memory/context_builder.py`：上下文生成策略。
- `src/multi_agent_asr/api/app.py`：HTTP API。
- `docs/architecture.md`：架构与扩展约束。
