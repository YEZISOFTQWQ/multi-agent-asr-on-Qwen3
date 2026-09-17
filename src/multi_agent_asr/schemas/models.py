from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TimeStamp(BaseModel):
    text: str
    start_time: float
    end_time: float


class AudioInfo(BaseModel):
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int


class SpeakerObservation(BaseModel):
    speaker_id: str | None = None
    confidence: float = 0.0
    source: str = "unknown"


class SceneObservation(BaseModel):
    label: str | None = None
    confidence: float = 0.0
    source: str = "unknown"


class SpeakerProfile(BaseModel):
    speaker_id: str
    display_name: str | None = None
    accent: str | None = None
    accent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    frequent_terms: list[str] = Field(default_factory=list)
    corrections: dict[str, str] = Field(default_factory=dict)
    environments: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class TranscriptionInput(BaseModel):
    audio_path: str
    session_id: str = "default"
    speaker_hint: str | None = None
    scene_hint: str | None = None
    language: str | None = None
    explicit_context: str | None = None
    return_time_stamps: bool = False


class TranscriptCandidate(BaseModel):
    text: str
    language: str | None = None
    context_used: str = ""
    time_stamps: list[TimeStamp] = Field(default_factory=list)


class VerificationResult(BaseModel):
    verified: bool
    warnings: list[str] = Field(default_factory=list)
    retry_context: str | None = None


class ASRResult(BaseModel):
    text: str
    language: str | None = None
    session_id: str
    speaker_id: str | None = None
    scene: str | None = None
    verified: bool
    warnings: list[str] = Field(default_factory=list)
    context_used: str = ""
    time_stamps: list[TimeStamp] = Field(default_factory=list)
