from __future__ import annotations

from multi_agent_asr.schemas import SceneObservation, SpeakerProfile


class ContextBuilder:
    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def build(
        self,
        *,
        profile: SpeakerProfile | None,
        scene: SceneObservation,
        recent_utterances: list[str],
        explicit_context: str | None = None,
    ) -> str:
        sections: list[str] = [
            "请以音频内容为主要依据。上下文仅用于消除歧义，不要添加音频中没有的信息。"
        ]

        if explicit_context and explicit_context.strip():
            sections.append(f"当前任务补充信息：{explicit_context.strip()}")

        if profile is not None:
            identity = profile.display_name or profile.speaker_id
            sections.append(f"说话人：{identity}。")
            if profile.accent and profile.accent_confidence >= 0.5:
                sections.append(
                    f"历史口音信息：{profile.accent}，置信度 {profile.accent_confidence:.2f}。"
                )
            if profile.frequent_terms:
                sections.append("常用术语：" + "、".join(profile.frequent_terms) + "。")
            if profile.corrections:
                pairs = [
                    f"“{wrong}”通常指“{right}”" for wrong, right in profile.corrections.items()
                ]
                sections.append("历史纠错：" + "；".join(pairs) + "。")

        if scene.label and scene.label != "unknown":
            sections.append(f"当前录音环境：{scene.label}，置信度 {scene.confidence:.2f}。")

        if recent_utterances:
            sections.append("最近对话：" + " ".join(recent_utterances) + "。")

        context = "\n".join(sections)
        return context[: self.max_chars]
