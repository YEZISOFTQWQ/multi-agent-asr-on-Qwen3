"""验证 LangGraph 编排、重试、Checkpoint 和节点记录。"""

import sqlite3
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import soundfile as sf

from multi_agent_asr.agents.audio_agent import AudioAgent
from multi_agent_asr.agents.memory_agent import MemoryAgent
from multi_agent_asr.agents.orchestrator import ASROrchestrator
from multi_agent_asr.agents.profile_update_agent import ProfileUpdateAgent
from multi_agent_asr.agents.qwen_asr_agent import QwenASRAgent
from multi_agent_asr.agents.scene_agent import SceneAgent
from multi_agent_asr.agents.speaker_agent import SpeakerAgent
from multi_agent_asr.agents.terminology_agent import TerminologyAgent
from multi_agent_asr.agents.verifier_agent import VerifierAgent
from multi_agent_asr.memory import ContextBuilder, SqliteMemoryRepository
from multi_agent_asr.schemas import SpeakerProfile, TranscriptCandidate, TranscriptionInput


class FakeQwenService:
    """返回固定文本，用于验证正常编排和术语纠错。"""

    async def transcribe(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        """返回包含待纠错术语的固定候选。"""
        del audio_path, return_time_stamps
        assert "四川口音" in context
        assert "Qwen3-ASR" in context
        assert "汽车驾驶舱" in context
        return TranscriptCandidate(
            text="我们继续讨论千问三。",
            language=language or "Chinese",
            context_used=context,
        )


class RetryQwenService:
    """第一次返回空文本，第二次成功，用于验证条件重试。"""

    def __init__(self) -> None:
        """初始化用于区分两次识别的调用计数。"""
        self.calls = 0

    async def transcribe(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        """第一次制造校验失败，随后确认重试上下文并成功。"""
        del audio_path, return_time_stamps
        self.calls += 1
        if self.calls == 1:
            return TranscriptCandidate(text="", language=language, context_used=context)
        assert "empty_transcript" in context
        return TranscriptCandidate(text="第二次识别成功", language=language, context_used=context)


def make_orchestrator(
    tmp_path: Path,
    service: object,
    *,
    max_retries: int = 1,
) -> tuple[ASROrchestrator, SqliteMemoryRepository]:
    """为每个测试创建使用临时 SQLite 文件的编排器。"""
    repository = SqliteMemoryRepository(tmp_path / "memory.sqlite3")
    memory_agent = MemoryAgent(repository, ContextBuilder(), recent_limit=5)
    orchestrator = ASROrchestrator(
        audio_agent=AudioAgent(),
        speaker_agent=SpeakerAgent(),
        scene_agent=SceneAgent(),
        memory_agent=memory_agent,
        asr_agent=QwenASRAgent(service),  # type: ignore[arg-type]
        terminology_agent=TerminologyAgent(),
        verifier_agent=VerifierAgent(),
        profile_update_agent=ProfileUpdateAgent(repository),
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
        max_retries=max_retries,
    )
    return orchestrator, repository


def write_silence(path: Path) -> None:
    """生成一个可被 SoundFile 正常解析的短静音 WAV。"""
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)


async def test_orchestrator_builds_context_and_records_verified_result(tmp_path: Path) -> None:
    """确认正常图路径会纠错、持久化并记录全部节点。"""
    audio_path = tmp_path / "sample.wav"
    write_silence(audio_path)
    orchestrator, repository = make_orchestrator(tmp_path, FakeQwenService())

    await orchestrator.initialize()
    try:
        await orchestrator.memory_agent.upsert_profile(
            SpeakerProfile(
                speaker_id="speaker_001",
                accent="四川口音",
                accent_confidence=0.8,
                frequent_terms=["Qwen3-ASR"],
                corrections={"千问三": "Qwen3-ASR"},
            )
        )

        result = await orchestrator.transcribe(
            TranscriptionInput(
                audio_path=str(audio_path),
                session_id="session-1",
                speaker_hint="speaker_001",
                scene_hint="汽车驾驶舱",
                language="Chinese",
            )
        )

        assert result.verified is True
        assert result.run_id
        assert result.speaker_id == "speaker_001"
        assert result.text == "我们继续讨论Qwen3-ASR。"
        assert result.applied_corrections[0].original == "千问三"
        assert result.applied_corrections[0].replacement == "Qwen3-ASR"
        assert await repository.recent_utterances("session-1", limit=5) == [result.text]

        records = await orchestrator.list_node_runs(result.run_id)
        assert {record.node_name for record in records} == {
            "audio",
            "speaker",
            "scene",
            "context",
            "asr",
            "terminology",
            "verify",
            "finalize",
            "persist",
        }
        assert all(record.status == "succeeded" for record in records)
    finally:
        await orchestrator.close()


async def test_graph_retries_and_persists_checkpoint(tmp_path: Path) -> None:
    """确认空结果触发第二次 ASR 并产生 Checkpoint。"""
    audio_path = tmp_path / "retry.wav"
    write_silence(audio_path)
    service = RetryQwenService()
    orchestrator, _ = make_orchestrator(tmp_path, service)

    await orchestrator.initialize()
    try:
        result = await orchestrator.transcribe(
            TranscriptionInput(
                audio_path=str(audio_path),
                session_id="retry-session",
                language="Chinese",
            )
        )
        records = await orchestrator.list_node_runs(result.run_id)
    finally:
        await orchestrator.close()

    assert service.calls == 2
    assert result.verified is True
    assert result.text == "第二次识别成功"
    assert [record.attempt for record in records if record.node_name == "asr"] == [1, 2]
    assert [record.attempt for record in records if record.node_name == "verify"] == [1, 2]
    assert [record.node_name for record in records].count("retry") == 1

    thread_id = f"retry-session:{result.run_id}"
    # 直接检查 LangGraph 自建表，避免仅凭返回值间接推断 Checkpoint 已写入。
    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        checkpoint_count = connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()[0]
    assert checkpoint_count > 0


async def test_failed_graph_node_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认节点异常会保存 failed 状态和错误摘要。"""
    run_uuid = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    monkeypatch.setattr("multi_agent_asr.agents.orchestrator.uuid4", lambda: run_uuid)
    orchestrator, _ = make_orchestrator(tmp_path, FakeQwenService())

    await orchestrator.initialize()
    try:
        with pytest.raises(FileNotFoundError):
            await orchestrator.transcribe(
                TranscriptionInput(
                    audio_path=str(tmp_path / "missing.wav"),
                    session_id="failed-session",
                )
            )
        records = await orchestrator.list_node_runs(run_uuid.hex)
    finally:
        await orchestrator.close()

    audio_record = next(record for record in records if record.node_name == "audio")
    assert audio_record.status == "failed"
    assert audio_record.error is not None
    assert "FileNotFoundError" in audio_record.error
