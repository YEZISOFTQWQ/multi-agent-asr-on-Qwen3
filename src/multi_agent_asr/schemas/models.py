"""定义 API、Agent、图状态和持久化层共享的数据契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TimeStamp(BaseModel):
    """一个带起止秒数的对齐文本片段。"""

    text: str
    start_time: float
    end_time: float


class AudioInfo(BaseModel):
    """经过检查的音频文件元数据。"""

    path: str
    duration_seconds: float
    sample_rate: int
    channels: int


class SpeakerObservation(BaseModel):
    """当前音频中的说话人判断及其来源。"""

    speaker_id: str | None = None
    confidence: float = 0.0
    source: str = "unknown"


class SceneObservation(BaseModel):
    """当前录音环境的判断及其来源。"""

    label: str | None = None
    confidence: float = 0.0
    source: str = "unknown"


class SpeakerProfile(BaseModel):
    """跨会话保存的说话人偏好、口音和术语信息。"""

    speaker_id: str
    display_name: str | None = None
    accent: str | None = None
    accent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    frequent_terms: list[str] = Field(default_factory=list)
    corrections: dict[str, str] = Field(default_factory=dict)
    environments: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class AppliedCorrection(BaseModel):
    """一次确定性术语替换的审计信息。"""

    original: str
    replacement: str
    occurrences: int = Field(ge=1)


class TranscriptionInput(BaseModel):
    """API 和 CLI 共用的转写请求。"""

    audio_path: str
    session_id: str = "default"
    speaker_hint: str | None = None
    scene_hint: str | None = None
    language: str | None = None
    explicit_context: str | None = None
    return_time_stamps: bool = False


class TranscriptCandidate(BaseModel):
    """ASR 输出及后处理 Agent 共享的候选文本。"""

    text: str
    language: str | None = None
    context_used: str = ""
    time_stamps: list[TimeStamp] = Field(default_factory=list)
    applied_corrections: list[AppliedCorrection] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """校验结论、告警和可选的重试提示。"""

    verified: bool
    warnings: list[str] = Field(default_factory=list)
    retry_context: str | None = None


class ASRResult(BaseModel):
    """返回给调用方并写入会话历史的最终结果。"""

    text: str
    language: str | None = None
    session_id: str
    run_id: str = ""
    speaker_id: str | None = None
    scene: str | None = None
    verified: bool
    warnings: list[str] = Field(default_factory=list)
    context_used: str = ""
    time_stamps: list[TimeStamp] = Field(default_factory=list)
    applied_corrections: list[AppliedCorrection] = Field(default_factory=list)


class NodeRunRecord(BaseModel):
    """一个 LangGraph 节点执行的持久化审计记录。"""

    id: int
    run_id: str
    thread_id: str
    node_name: str
    status: Literal["running", "succeeded", "failed"]
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = None
    attempt: int = Field(ge=1)
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
