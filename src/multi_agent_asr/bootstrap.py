"""集中装配 Agent、服务、仓库和 LangGraph 编排器。"""

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
from multi_agent_asr.observability import SqliteRunRepository
from multi_agent_asr.services import QwenASRService


def build_orchestrator(settings: Settings) -> tuple[ASROrchestrator, QwenASRService]:
    """根据配置装配一套共享依赖并返回编排器与模型服务。"""
    settings.ensure_runtime_directories()
    # 业务记忆与节点审计共享一个 SQLite 文件，但由不同仓库维护各自表。
    repository = SqliteMemoryRepository(settings.database_path)
    run_repository = SqliteRunRepository(settings.database_path)
    # 所有请求共享同一个服务实例，避免在同一进程重复加载模型权重。
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
        checkpoint_database_path=settings.checkpoint_database_path,
        run_repository=run_repository,
        max_retries=settings.max_asr_retries,
    )
    return orchestrator, qwen_service
