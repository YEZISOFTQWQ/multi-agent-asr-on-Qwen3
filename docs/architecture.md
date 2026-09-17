# 架构说明

## 数据流

```text
音频请求
  -> AudioAgent 检查音频
  -> SpeakerAgent 获取说话人线索
  -> SceneAgent 获取环境线索
  -> MemoryAgent 查询画像和会话历史
  -> ContextBuilder 生成受长度约束的 context
  -> QwenASRAgent 调用 Qwen3-ASR
  -> TerminologyAgent 应用已确认的精确术语纠错
  -> VerifierAgent 检查结果
  -> ProfileUpdateAgent 写入可信历史
```

音频、说话人和场景分析可以并行执行。模型推理使用进程内异步锁串行进入单个模型实例，避免同一张 GPU 上重复加载权重。

## 记忆分层

1. `speaker_profiles`：长期画像、常用术语和确认过的纠错。
2. `utterances`：按 `session_id` 保存最近对话，作为短期记忆。
3. 场景信息属于当前观测，只有经过独立策略确认后才应写入长期画像。

## 当前边界

- SpeakerAgent 的基础实现只使用调用方传入的 `speaker_hint`。
- SceneAgent 的基础实现只使用调用方传入的 `scene_hint`。
- VerifierAgent 执行空结果、控制字符和异常重复检查。
- TerminologyAgent 只读取画像中的显式纠错字典，使用单次最长匹配替换并记录修改明细。
- Qwen3-ASR 是唯一会加载大模型的组件，并且按需加载。

## 后续扩展

- 在 `services/` 中增加 pyannote 或 SpeechBrain 适配器，并注入 SpeakerAgent。
- 增加音频场景和口音分类器，输出标签、置信度和模型版本。
- 使用 Qwen3 ForcedAligner 提供音字对齐证据。
- 对低可信片段生成无上下文与定向上下文两个候选，再由证据规则选择。
- 流式模式中分别维护稳定前缀和可回滚尾部。
