from __future__ import annotations

from multi_agent_asr.agents.audio_agent import AudioAgent
from multi_agent_asr.agents.memory_agent import MemoryAgent
from multi_agent_asr.agents.orchestrator import ASROrchestrator
from multi_agent_asr.agents.profile_update_agent import ProfileUpdateAgent
from multi_agent_asr.agents.qwen_asr_agent import QwenASRAgent
from multi_agent_asr.agents.scene_agent import SceneAgent
from multi_agent_asr.agents.speaker_agent import SpeakerAgent
from multi_agent_asr.agents.terminology_agent import TerminologyAgent
from multi_agent_asr.agents.verifier_agent import VerifierAgent
from multi_agent_asr.config import Settings
from multi_agent_asr.memory import ContextBuilder, SqliteMemoryRepository
from multi_agent_asr.services import QwenASRService


def build_orchestrator(settings: Settings) -> tuple[ASROrchestrator, QwenASRService]:
    settings.ensure_runtime_directories()
    repository = SqliteMemoryRepository(settings.database_path)
    qwen_service = QwenASRService(settings)
    memory_agent = MemoryAgent(
        repository=repository,
        context_builder=ContextBuilder(settings.max_context_chars),
        recent_limit=settings.recent_utterance_limit,
    )
    orchestrator = ASROrchestrator(
        audio_agent=AudioAgent(),
        speaker_agent=SpeakerAgent(),
        scene_agent=SceneAgent(),
        memory_agent=memory_agent,
        asr_agent=QwenASRAgent(qwen_service),
        terminology_agent=TerminologyAgent(),
        verifier_agent=VerifierAgent(),
        profile_update_agent=ProfileUpdateAgent(repository),
    )
    return orchestrator, qwen_service
