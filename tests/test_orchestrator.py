from pathlib import Path

import numpy as np
import soundfile as sf

from multi_agent_asr.agents.audio_agent import AudioAgent
from multi_agent_asr.agents.memory_agent import MemoryAgent
from multi_agent_asr.agents.orchestrator import ASROrchestrator
from multi_agent_asr.agents.profile_update_agent import ProfileUpdateAgent
from multi_agent_asr.agents.qwen_asr_agent import QwenASRAgent
from multi_agent_asr.agents.scene_agent import SceneAgent
from multi_agent_asr.agents.speaker_agent import SpeakerAgent
from multi_agent_asr.agents.verifier_agent import VerifierAgent
from multi_agent_asr.memory import ContextBuilder, SqliteMemoryRepository
from multi_agent_asr.schemas import SpeakerProfile, TranscriptCandidate, TranscriptionInput


class FakeQwenService:
    async def transcribe(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        del audio_path, return_time_stamps
        assert "四川口音" in context
        assert "Qwen3-ASR" in context
        assert "汽车驾驶舱" in context
        return TranscriptCandidate(
            text="我们继续讨论 Qwen3-ASR。",
            language=language or "Chinese",
            context_used=context,
        )


async def test_orchestrator_builds_context_and_records_verified_result(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)

    repository = SqliteMemoryRepository(tmp_path / "memory.sqlite3")
    memory_agent = MemoryAgent(repository, ContextBuilder(), recent_limit=5)
    orchestrator = ASROrchestrator(
        audio_agent=AudioAgent(),
        speaker_agent=SpeakerAgent(),
        scene_agent=SceneAgent(),
        memory_agent=memory_agent,
        asr_agent=QwenASRAgent(FakeQwenService()),  # type: ignore[arg-type]
        verifier_agent=VerifierAgent(),
        profile_update_agent=ProfileUpdateAgent(repository),
    )
    await orchestrator.initialize()
    await memory_agent.upsert_profile(
        SpeakerProfile(
            speaker_id="speaker_001",
            accent="四川口音",
            accent_confidence=0.8,
            frequent_terms=["Qwen3-ASR"],
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
    assert result.speaker_id == "speaker_001"
    assert await repository.recent_utterances("session-1", limit=5) == [result.text]
