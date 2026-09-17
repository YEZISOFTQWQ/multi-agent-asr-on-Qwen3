"""验证上下文内容、安全提示和字符预算。"""

from multi_agent_asr.memory import ContextBuilder
from multi_agent_asr.schemas import SceneObservation, SpeakerProfile


def test_context_contains_profile_scene_history_and_safety_instruction() -> None:
    """确认各类线索和音频优先约束都进入上下文。"""
    profile = SpeakerProfile(
        speaker_id="speaker_001",
        display_name="张三",
        accent="四川口音",
        accent_confidence=0.81,
        frequent_terms=["Qwen3-ASR", "ForcedAligner"],
        corrections={"福斯阿莱纳": "ForcedAligner"},
    )

    context = ContextBuilder(max_chars=2000).build(
        profile=profile,
        scene=SceneObservation(label="汽车驾驶舱", confidence=0.9, source="test"),
        recent_utterances=["我们继续讨论流式语音识别"],
        explicit_context="当前章节是上下文偏置",
    )

    assert "音频内容为主要依据" in context
    assert "四川口音" in context
    assert "Qwen3-ASR" in context
    assert "ForcedAligner" in context
    assert "汽车驾驶舱" in context
    assert "流式语音识别" in context


def test_context_respects_character_budget() -> None:
    """确认过长上下文会被严格截断。"""
    context = ContextBuilder(max_chars=120).build(
        profile=None,
        scene=SceneObservation(label="unknown"),
        recent_utterances=["很长的历史" * 100],
    )
    assert len(context) == 120
