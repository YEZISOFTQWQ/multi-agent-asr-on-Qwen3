"""把画像、场景和会话历史整理为安全的 ASR 上下文。"""

from __future__ import annotations

from multi_agent_asr.schemas import SceneObservation, SpeakerProfile


class ContextBuilder:
    """按可信度和字符预算构建仅用于消歧的模型上下文。"""

    def __init__(self, max_chars: int = 2000) -> None:
        """设置传给模型的最大上下文字符数。"""
        self.max_chars = max_chars

    def build(
        self,
        *,
        profile: SpeakerProfile | None,
        scene: SceneObservation,
        recent_utterances: list[str],
        explicit_context: str | None = None,
    ) -> str:
        """按固定优先级组合可用线索，并严格截断到字符预算。"""
        # 安全约束始终放在首段，所有个性化信息都只能帮助消歧。
        sections: list[str] = [
            "请以音频内容为主要依据。上下文仅用于消除歧义，不要添加音频中没有的信息。"
        ]

        if explicit_context and explicit_context.strip():
            sections.append(f"当前任务补充信息：{explicit_context.strip()}")

        if profile is not None:
            identity = profile.display_name or profile.speaker_id
            sections.append(f"说话人：{identity}。")
            # 低置信度口音可能误导识别，因此不进入模型上下文。
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
        # 最终硬截断保证任何来源都不能突破模型上下文预算。
        return context[: self.max_chars]
