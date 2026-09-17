"""验证 SQLite 画像和已校验会话历史的持久化。"""

from pathlib import Path

from multi_agent_asr.memory import SqliteMemoryRepository
from multi_agent_asr.schemas import ASRResult, SpeakerProfile


async def test_profile_and_verified_history_round_trip(tmp_path: Path) -> None:
    """确认画像可往返，并且上下文历史只读取已验证文本。"""
    repository = SqliteMemoryRepository(tmp_path / "memory.sqlite3")
    await repository.initialize()

    saved = await repository.upsert_profile(
        SpeakerProfile(
            speaker_id="speaker_001",
            accent="四川口音",
            accent_confidence=0.8,
            frequent_terms=["Qwen3-ASR"],
        )
    )
    loaded = await repository.get_profile("speaker_001")

    assert loaded is not None
    assert loaded.accent == "四川口音"
    assert loaded.frequent_terms == ["Qwen3-ASR"]
    assert saved.updated_at is not None

    await repository.append_utterance(
        ASRResult(
            text="第一句",
            language="Chinese",
            session_id="session-1",
            verified=True,
        )
    )
    await repository.append_utterance(
        ASRResult(
            text="未通过校验",
            language="Chinese",
            session_id="session-1",
            verified=False,
        )
    )

    assert await repository.recent_utterances("session-1", limit=5) == ["第一句"]
