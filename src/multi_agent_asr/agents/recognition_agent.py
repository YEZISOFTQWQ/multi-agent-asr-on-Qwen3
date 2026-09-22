"""实现识别策略选择、模型调用、术语工具调用和候选自评。"""

from __future__ import annotations

from typing import Protocol

from multi_agent_asr.schemas import (
    AgentTraceEntry,
    AudioInfo,
    RecognitionResult,
    SpeakerProfile,
    TranscriptCandidate,
    TranscriptionInput,
)
from multi_agent_asr.tools import TerminologyCorrector


class TranscriptionService(Protocol):
    """Recognition Agent 依赖的模型服务协议。"""

    async def transcribe(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        """执行一次模型转写。"""


class RecognitionAgent:
    """选择识别策略并产出供审核的候选文本。

    Agent 根据可信上下文和上一轮审核反馈决定如何调用模型。模型输出后，
    再调用确定性术语工具应用已经确认的说话人纠错规则。
    """

    def __init__(
        self,
        service: TranscriptionService,
        terminology_corrector: TerminologyCorrector,
    ) -> None:
        """注入模型服务和确定性术语纠错工具。

        Args:
            service: 满足转写协议的模型服务，生产环境使用 Qwen3-ASR。
            terminology_corrector: 对候选执行非级联替换的工具。
        """
        self.service = service
        self.terminology_corrector = terminology_corrector

    async def run(
        self,
        *,
        request: TranscriptionInput,
        audio: AudioInfo,
        context: str,
        profile: SpeakerProfile | None,
        attempt: int,
        review_feedback: str | None,
    ) -> RecognitionResult:
        """选择识别动作，调用模型和纠错工具，再评估候选。

        Args:
            request: 原始转写请求，用于读取语言和时间戳选项。
            audio: 已通过 Audio Preparation Agent 检查的音频。
            context: Context Selection Agent 生成的可信上下文。
            profile: 已通过身份置信度筛选的说话人画像。
            attempt: 当前识别尝试序号，从 1 开始。
            review_feedback: 上一轮审核生成的问题摘要。

        Returns:
            候选文本、本轮动作、初步评估和审计轨迹。
        """
        # 审核反馈代表上一候选存在具体问题，因此优先级高于普通上下文；
        # 首次识别则根据是否存在可信上下文选择有上下文或裸转写。
        if review_feedback:
            action = "retry_with_feedback"
            reason = "previous_candidate_failed_review"
            effective_context = (
                f"{context}\n重试提示：请避免以下问题：{review_feedback}。"
                "音频证据仍具有最高优先级。"
            ).strip()
        elif context.strip():
            action = "transcribe_with_context"
            reason = "trusted_context_is_available"
            effective_context = context
        else:
            action = "transcribe_without_context"
            reason = "no_trusted_context_is_available"
            effective_context = ""

        candidate = await self.service.transcribe(
            audio_path=audio.path,
            context=effective_context,
            language=request.language,
            return_time_stamps=request.return_time_stamps,
        )
        # 术语替换在模型调用之后执行，且只使用经过筛选的画像，避免把
        # 低可信身份信息写入候选。
        candidate = await self.terminology_corrector.apply(candidate, profile)
        assessment = "candidate_ready" if candidate.text.strip() else "empty_candidate"
        trace = [
            AgentTraceEntry(
                step=1,
                observation={
                    "attempt": attempt,
                    "has_context": bool(context.strip()),
                    "has_review_feedback": bool(review_feedback),
                    "language": request.language,
                },
                action=action,
                reason=reason,
                outcome=assessment,
            )
        ]
        return RecognitionResult(
            candidate=candidate,
            action=action,
            attempt=attempt,
            assessment=assessment,
            trace=trace,
        )
