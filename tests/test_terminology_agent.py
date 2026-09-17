"""验证术语纠错的最长匹配和非级联行为。"""

from multi_agent_asr.agents.terminology_agent import TerminologyAgent
from multi_agent_asr.schemas import SpeakerProfile, TranscriptCandidate


async def test_applies_longest_confirmed_corrections_without_cascading() -> None:
    """确认最长匹配优先且替换结果不会再次参与替换。"""
    candidate = TranscriptCandidate(text="千问三和千问三")
    profile = SpeakerProfile(
        speaker_id="speaker_001",
        corrections={
            "千问": "Qwen",
            "千问三": "Qwen3-ASR",
            "Qwen3-ASR": "不应二次替换",
        },
    )

    corrected = await TerminologyAgent().apply(candidate, profile)

    assert candidate.text == "千问三和千问三"
    assert corrected.text == "Qwen3-ASR和Qwen3-ASR"
    assert len(corrected.applied_corrections) == 1
    assert corrected.applied_corrections[0].original == "千问三"
    assert corrected.applied_corrections[0].occurrences == 2


async def test_returns_original_candidate_without_profile() -> None:
    """确认没有画像时保持原候选对象。"""
    candidate = TranscriptCandidate(text="保持原文")

    corrected = await TerminologyAgent().apply(candidate, None)

    assert corrected is candidate
