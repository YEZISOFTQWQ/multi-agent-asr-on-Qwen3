"""定义 Supervisor 管理的一次 ASR 运行共享状态。"""

from __future__ import annotations

from typing import TypedDict

from multi_agent_asr.schemas import (
    ASRResult,
    AudioInfo,
    AudioPreparationResult,
    ContextSelectionResult,
    RecognitionResult,
    SceneObservation,
    SpeakerObservation,
    SpeakerProfile,
    SupervisorDecision,
    TranscriptCandidate,
    TranscriptionInput,
    TranscriptReviewResult,
    VerificationResult,
)


class ASRGraphState(TypedDict, total=False):
    """五个 Agent 通过 LangGraph 交换的强类型黑板状态。"""

    # 请求身份字段在一次图运行中保持不变。
    request: TranscriptionInput
    run_id: str
    thread_id: str

    # Supervisor 字段记录当前决策以及完整路由历史。
    supervisor_step: int
    supervisor_decision: SupervisorDecision
    supervisor_history: list[SupervisorDecision]

    # 各阶段同时保留完整结果和常用派生值，避免下游节点了解上游内部结构。
    audio_preparation: AudioPreparationResult | None
    audio: AudioInfo | None
    context_selection: ContextSelectionResult | None
    speaker: SpeakerObservation | None
    scene: SceneObservation | None
    profile: SpeakerProfile | None
    context: str | None
    recognition: RecognitionResult | None
    candidate: TranscriptCandidate | None
    review: TranscriptReviewResult | None
    verification: VerificationResult | None
    result: ASRResult

    # 重试预算由 Supervisor 更新，反馈由 Recognition Agent 消费。
    retry_count: int
    max_retries: int
    retry_feedback: str | None
