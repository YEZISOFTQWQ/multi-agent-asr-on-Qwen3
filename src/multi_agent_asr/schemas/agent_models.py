"""定义五个自治 Agent 的输入、决策、结果和审计轨迹。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import (
    AudioInfo,
    SceneObservation,
    SpeakerObservation,
    SpeakerProfile,
    TranscriptCandidate,
    VerificationResult,
)


class AgentTraceEntry(BaseModel):
    """Agent 一次观察、决策和行动的可审计摘要。"""

    step: int = Field(ge=1)
    observation: dict[str, Any] = Field(default_factory=dict)
    action: str
    reason: str
    outcome: str | None = None


AudioPreparationAction = Literal["accept", "downmix_to_mono", "reject"]


class AudioPreparationResult(BaseModel):
    """Audio Preparation Agent 的终态。"""

    accepted: bool
    audio: AudioInfo | None = None
    failure_reason: str | None = None
    actions: list[AudioPreparationAction] = Field(default_factory=list)
    trace: list[AgentTraceEntry] = Field(default_factory=list)


ContextStrategy = Literal["personalized", "session_only", "conservative", "budget_reduced"]


class ContextSelectionResult(BaseModel):
    """Context Selection Agent 选择证据并渲染上下文后的结果。"""

    context: str
    strategy: ContextStrategy
    speaker: SpeakerObservation
    scene: SceneObservation
    profile: SpeakerProfile | None = None
    included_sources: list[str] = Field(default_factory=list)
    excluded_sources: list[str] = Field(default_factory=list)
    trace: list[AgentTraceEntry] = Field(default_factory=list)


RecognitionAction = Literal[
    "transcribe_with_context",
    "transcribe_without_context",
    "retry_with_feedback",
]


class RecognitionResult(BaseModel):
    """Recognition Agent 一次决策周期生成的候选。"""

    candidate: TranscriptCandidate
    action: RecognitionAction
    attempt: int = Field(ge=1)
    assessment: Literal["candidate_ready", "empty_candidate"]
    trace: list[AgentTraceEntry] = Field(default_factory=list)


ReviewDecision = Literal["accept", "retry_recognition", "finalize_unverified"]


class TranscriptReviewResult(BaseModel):
    """Transcript Review Agent 的审查结论和下一步反馈。"""

    verification: VerificationResult
    decision: ReviewDecision
    feedback: str | None = None
    checks_run: list[str] = Field(default_factory=list)
    attempt: int = Field(ge=1)
    trace: list[AgentTraceEntry] = Field(default_factory=list)


class SupervisorObservation(BaseModel):
    """Supervisor 做路由决策时看到的最小全局状态。"""

    has_audio: bool
    has_context: bool
    has_candidate: bool
    has_review: bool
    review_decision: ReviewDecision | None = None
    review_feedback: str | None = None
    retry_count: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    step_count: int = Field(ge=0)


SupervisorAction = Literal[
    "audio_preparation",
    "context_selection",
    "recognition",
    "transcript_review",
    "finalize",
]


class SupervisorDecision(BaseModel):
    """Supervisor 为 LangGraph 选择的下一项动作。"""

    action: SupervisorAction
    reason: str
    step: int = Field(ge=1)
    retry_count: int = Field(ge=0)
    retry_feedback: str | None = None
    clears_previous_candidate: bool = False
