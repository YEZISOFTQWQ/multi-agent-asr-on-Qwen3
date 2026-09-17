from __future__ import annotations

from multi_agent_asr.schemas import SceneObservation


class SceneAgent:
    """Baseline scene agent; replace with an acoustic scene classifier later."""

    async def analyze(self, audio_path: str, scene_hint: str | None) -> SceneObservation:
        del audio_path
        if scene_hint:
            return SceneObservation(label=scene_hint, confidence=1.0, source="hint")
        return SceneObservation(label="unknown")
