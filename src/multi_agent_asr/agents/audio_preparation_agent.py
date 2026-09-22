"""实现有界的音频观察、决策、处理和复检循环。"""

from __future__ import annotations

from pathlib import Path

from multi_agent_asr.schemas import AgentTraceEntry, AudioPreparationResult
from multi_agent_asr.tools import AudioPreparationTools


class AudioPreparationAgent:
    """把输入音频处理到 ASR 可以稳定消费的状态。

    Agent 每轮先观察音频元数据，再从接受、拒绝和声道混合三个动作中
    选择一个。产生新文件后会重新观察，而不会直接假设处理结果有效。
    `max_steps` 为观察与动作循环提供硬上限。
    """

    def __init__(
        self,
        tools: AudioPreparationTools,
        processed_root: Path,
        max_steps: int = 3,
    ) -> None:
        """配置音频工具、处理文件目录和最大决策步数。

        Args:
            tools: Agent 可以调用的音频检查和变换工具。
            processed_root: 保存本次运行生成音频的根目录。
            max_steps: 单次运行允许的最大观察与动作次数。
        """
        self.tools = tools
        self.processed_root = processed_root
        # 至少保留一次观察机会，避免配置错误让 Agent 无条件失败。
        self.max_steps = max(1, max_steps)

    async def run(self, audio_path: str, run_id: str) -> AudioPreparationResult:
        """反复观察并处理音频，直到接受、拒绝或耗尽步数。

        Args:
            audio_path: 用户提交的音频路径。
            run_id: 当前运行的唯一标识，用于隔离中间文件。

        Returns:
            最终音频状态、已执行动作和逐步审计轨迹。
        """
        current_path = audio_path
        actions = []
        trace: list[AgentTraceEntry] = []

        for step in range(1, self.max_steps + 1):
            observation = await self.tools.inspect(current_path)
            observed = {
                "path": observation.path,
                "duration_seconds": observation.duration_seconds,
                "sample_rate": observation.sample_rate,
                "channels": observation.channels,
            }

            # 先排除无法恢复的元数据错误，再处理可以通过声道混合修复的
            # 多声道输入，保证每轮只执行一个含义明确的动作。
            if observation.duration_seconds <= 0:
                action = "reject"
                reason = "audio_has_no_frames"
            elif observation.sample_rate <= 0 or observation.channels <= 0:
                action = "reject"
                reason = "invalid_audio_metadata"
            elif observation.channels > 1:
                action = "downmix_to_mono"
                reason = "multi_channel_audio_requires_deterministic_downmix"
            else:
                action = "accept"
                reason = "audio_is_ready_for_recognition"

            actions.append(action)
            if action == "accept":
                trace.append(
                    AgentTraceEntry(
                        step=step,
                        observation=observed,
                        action=action,
                        reason=reason,
                        outcome="accepted",
                    )
                )
                return AudioPreparationResult(
                    accepted=True,
                    audio=observation,
                    actions=actions,
                    trace=trace,
                )

            if action == "reject":
                trace.append(
                    AgentTraceEntry(
                        step=step,
                        observation=observed,
                        action=action,
                        reason=reason,
                        outcome="rejected",
                    )
                )
                return AudioPreparationResult(
                    accepted=False,
                    failure_reason=reason,
                    actions=actions,
                    trace=trace,
                )

            # 中间文件按 run_id 隔离；同一次运行重试时覆盖固定文件名，
            # 避免为每个观察步骤留下内容相同的副本。
            output_path = self.processed_root / run_id / "prepared.wav"
            prepared = await self.tools.downmix_to_mono(current_path, str(output_path))
            trace.append(
                AgentTraceEntry(
                    step=step,
                    observation=observed,
                    action=action,
                    reason=reason,
                    outcome=f"created:{prepared.path}",
                )
            )
            current_path = prepared.path

        # 只有动作循环未得到可接受音频时才会到达这里。
        return AudioPreparationResult(
            accepted=False,
            failure_reason="audio_preparation_step_limit_exceeded",
            actions=actions,
            trace=trace,
        )
