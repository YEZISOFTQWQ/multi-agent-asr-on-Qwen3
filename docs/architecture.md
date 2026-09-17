# 架构说明

## 编排框架

`ASROrchestrator` 使用 LangGraph 1.x 编译并执行 `StateGraph`。它负责创建唯一的 `run_id` 和 `thread_id`、管理 `AsyncSqliteSaver` 的生命周期，并保留原有的 `transcribe()` 接口。Agent 的业务职责保持独立，LangGraph 只负责节点顺序、并行汇合、条件分支和状态持久化。

LangGraph 不要求大模型 API。当前所有节点调用本地 Python 组件，ASR 节点按需加载本地 Qwen3-ASR。

## 状态图

```text
START
 ├─ audio
 ├─ speaker
 └─ scene
      ↓ 三个节点全部完成
 context
      ↓
 asr → terminology → verify
                       ├─ verified / 达到上限 → finalize → persist → END
                       └─ 可重试 → retry ───────────────→ asr
```

- `audio`、`speaker` 和 `scene` 并行运行，并写入不同的 `ASRGraphState` 字段。
- `context` 查询说话人画像与会话历史，并生成受长度约束的上下文。
- `retry` 增加 `retry_count`，把校验原因写入定向提示，然后回到 `asr`。
- `MASR_MAX_ASR_RETRIES` 限制额外识别次数；默认值 `1` 代表最多识别两次。
- `finalize` 生成带 `run_id` 的 `ASRResult`，`persist` 写入会话历史。

模型推理使用进程内异步锁串行进入单个模型实例，避免同一张 GPU 重复加载权重。

## 状态与可观测性

1. `speaker_profiles`：长期画像、常用术语和确认过的纠错。
2. `utterances`：按 `session_id` 保存最近对话；后续上下文只读取已通过校验的文本。
3. `node_runs`：按 `run_id` 记录节点状态、开始与结束时间、耗时、attempt 和错误。
4. `checkpoints` 与 `checkpoint_blobs`：由 LangGraph 管理，用于保存图状态。

业务记忆和节点记录默认位于 `memory.sqlite3`。LangGraph Checkpoint 单独位于 `checkpoints.sqlite3`。每次请求使用 `{session_id}:{run_id}` 作为 Checkpoint `thread_id`，避免同一会话中的新请求续接旧图。

## 当前边界

- `SpeakerAgent` 的基础实现只使用调用方传入的 `speaker_hint`。
- `SceneAgent` 的基础实现只使用调用方传入的 `scene_hint`。
- `VerifierAgent` 执行空结果、控制字符和异常重复检查。
- `TerminologyAgent` 只读取画像中的显式纠错字典，使用单次最长匹配替换并记录修改明细。
- Qwen3-ASR 是唯一会加载大模型的组件，并且按需加载。
- Checkpoint 保存执行状态；`node_runs` 保存面向开发者的运行审计，两者用途不同。

## 后续扩展

- 在 `services/` 中增加 pyannote 或 SpeechBrain 适配器，并注入 `SpeakerAgent`。
- 增加音频场景和口音分类器，输出标签、置信度和模型版本。
- 使用 Qwen3 ForcedAligner 提供音字对齐证据。
- 对低可信片段生成无上下文与定向上下文两个候选，再由证据规则选择。
- 流式模式中分别维护稳定前缀和可回滚尾部。
