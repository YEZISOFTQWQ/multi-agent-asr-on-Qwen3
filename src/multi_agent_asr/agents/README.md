# 五个 Agent 实现说明

本目录只保存具有目标、观察、动作选择、反馈处理和终止约束的控制器，以及管理 LangGraph 资源的 `ASROrchestrator`。固定能力已经移动到 `src/multi_agent_asr/tools/` 和 `services/`。

完整设计、时序、配置、失败策略和测试证据见 [`docs/agent-architecture.md`](../../../docs/agent-architecture.md)。

## Agent 判定标准

本项目只有同时满足以下条件的实现才称为 Agent：

1. 有明确目标。
2. 观察结构化状态。
3. 可以在多个动作之间选择。
4. 有显式决策策略和选择原因。
5. 行动结果会影响下一次决策。
6. 有最大步数、重试预算或终止条件。

LangGraph 节点、数据库仓库、模型适配器和固定函数本身不属于 Agent。

## SupervisorAgent

文件：`supervisor_agent.py`

目标是在预算内补齐音频、上下文、候选和审查结果。它观察 `SupervisorObservation`，从以下动作中选择一个：

- `audio_preparation`
- `context_selection`
- `recognition`
- `transcript_review`
- `finalize`

Review 请求重试时，Supervisor 增加重试计数，保存反馈，清除旧候选和旧审查结果，再选择 Recognition。`max_steps` 保证全局循环有界。

## AudioPreparationAgent

文件：`audio_preparation_agent.py`

目标是得到元数据有效的单声道音频。它调用 `AudioPreparationTools` 观察音频，并选择：

- `accept`
- `downmix_to_mono`
- `reject`

多声道音频转换后会再次检查，因此具有独立反馈循环。处理结果包含全部动作和 `AgentTraceEntry`。

## ContextSelectionAgent

文件：`context_selection_agent.py`

目标是在可信度和字符预算内选择消歧证据。它观察说话人提示、场景提示、画像、已验证历史和用户上下文，选择：

- `personalized`
- `session_only`
- `conservative`
- `budget_reduced`

如果首次渲染触及预算且包含历史，Agent 会删除历史后重新渲染。画像只有在身份置信度达到阈值时才能使用。

## RecognitionAgent

文件：`recognition_agent.py`

目标是根据音频、上下文和 Review 反馈生成候选。动作包括：

- `transcribe_with_context`
- `transcribe_without_context`
- `retry_with_feedback`

Agent 调用共享 `QwenASRService`，随后调用确定性的 `TerminologyCorrector`。Review 反馈会改变下一轮动作和有效上下文。

## TranscriptReviewAgent

文件：`transcript_review_agent.py`

目标是选择适用检查并返回可执行结论。它始终执行文本检查；候选包含时间戳时才选择时间戳检查。动作包括：

- `accept`
- `retry_recognition`
- `finalize_unverified`

它只返回反馈，不直接修改候选。

## ASROrchestrator

文件：`orchestrator.py`

`ASROrchestrator` 是资源和生命周期控制器，不是 Agent。它负责：

- 初始化记忆库和运行记录库；
- 打开及关闭 `AsyncSqliteSaver`；
- 装配 Agent 图节点；
- 为每次请求创建 `run_id` 和 `thread_id`；
- 调用编译后的 LangGraph；
- 查询节点运行记录。

## 工具边界

| 工具或服务 | 用途 |
| --- | --- |
| `SoundFileAudioTools` | 音频检查、单声道转换 |
| `SpeakerResolver` | 显式说话人提示解析 |
| `SceneResolver` | 显式场景提示解析 |
| `ContextBuilder` | 上下文渲染和硬截断 |
| `QwenASRService` | 模型懒加载、并发控制和推理 |
| `TerminologyCorrector` | 已确认术语替换 |
| `TranscriptValidator` | 确定性质量检查 |
| `HistoryRecorder` | 写入会话历史 |

工具只执行请求的操作，不自主选择下一步。
