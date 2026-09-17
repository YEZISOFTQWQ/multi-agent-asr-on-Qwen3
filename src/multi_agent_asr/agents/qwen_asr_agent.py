from __future__ import annotations

from multi_agent_asr.schemas import TranscriptCandidate
from multi_agent_asr.services import QwenASRService


class QwenASRAgent:
    def __init__(self, service: QwenASRService) -> None:
        self.service = service

    async def transcribe(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        return await self.service.transcribe(
            audio_path=audio_path,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
