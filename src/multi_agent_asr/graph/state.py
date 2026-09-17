from __future__ import annotations

from typing import TypedDict

from multi_agent_asr.schemas import (
    ASRResult,
    AudioInfo,
    SceneObservation,
    SpeakerObservation,
    SpeakerProfile,
    TranscriptCandidate,
    TranscriptionInput,
    VerificationResult,
)


class ASRGraphState(TypedDict, total=False):
    request: TranscriptionInput
    run_id: str
    thread_id: str
    audio: AudioInfo
    speaker: SpeakerObservation
    scene: SceneObservation
    profile: SpeakerProfile | None
    context: str
    candidate: TranscriptCandidate
    verification: VerificationResult
    result: ASRResult
    retry_count: int
    max_retries: int
    retry_reason: str | None
