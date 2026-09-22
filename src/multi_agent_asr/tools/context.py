"""提供不带自治决策的说话人和场景线索解析工具。"""

from __future__ import annotations

from multi_agent_asr.schemas import SceneObservation, SpeakerObservation


class SpeakerResolver:
    """解析显式说话人提示；未知时保持未知。"""

    async def resolve(self, audio_path: str, speaker_hint: str | None) -> SpeakerObservation:
        """把可信的显式提示转换为结构化观察。"""
        # 当前实现只解析调用方提示，保留 audio_path 参数是为了让未来的
        # 声纹实现可以遵循同一工具接口。
        del audio_path
        if speaker_hint:
            return SpeakerObservation(speaker_id=speaker_hint, confidence=1.0, source="hint")
        return SpeakerObservation()


class SceneResolver:
    """解析显式场景提示；未知时返回 unknown。"""

    async def resolve(self, audio_path: str, scene_hint: str | None) -> SceneObservation:
        """把显式提示转换为结构化场景观察。"""
        # 当前实现不推断音频环境；未知值保持显式可见，便于调用方判断
        # 是否需要接入场景分类模型。
        del audio_path
        if scene_hint:
            return SceneObservation(label=scene_hint, confidence=1.0, source="hint")
        return SceneObservation(label="unknown")
