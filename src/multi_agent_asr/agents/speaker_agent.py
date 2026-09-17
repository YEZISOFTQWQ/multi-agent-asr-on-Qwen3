"""从音频或显式提示中获取说话人线索。"""

from __future__ import annotations

from multi_agent_asr.schemas import SpeakerObservation


class SpeakerAgent:
    """说话人识别基线；后续可替换为声纹嵌入服务。"""

    async def identify(self, audio_path: str, speaker_hint: str | None) -> SpeakerObservation:
        """优先采用显式说话人提示，否则返回未知说话人。"""
        del audio_path
        if speaker_hint:
            return SpeakerObservation(speaker_id=speaker_hint, confidence=1.0, source="hint")
        return SpeakerObservation()
