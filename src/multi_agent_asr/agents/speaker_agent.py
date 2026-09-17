from __future__ import annotations

from multi_agent_asr.schemas import SpeakerObservation


class SpeakerAgent:
    """Baseline speaker agent; replace with a speaker embedding service later."""

    async def identify(self, audio_path: str, speaker_hint: str | None) -> SpeakerObservation:
        del audio_path
        if speaker_hint:
            return SpeakerObservation(speaker_id=speaker_hint, confidence=1.0, source="hint")
        return SpeakerObservation()
