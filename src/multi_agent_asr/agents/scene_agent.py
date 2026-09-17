"""从音频或显式提示中获取录音场景线索。"""

from __future__ import annotations

from multi_agent_asr.schemas import SceneObservation


class SceneAgent:
    """录音场景基线；后续可替换为声学场景分类器。"""

    async def analyze(self, audio_path: str, scene_hint: str | None) -> SceneObservation:
        """优先采用显式场景提示，否则返回 unknown 基线结果。"""
        del audio_path
        if scene_hint:
            return SceneObservation(label=scene_hint, confidence=1.0, source="hint")
        return SceneObservation(label="unknown")
