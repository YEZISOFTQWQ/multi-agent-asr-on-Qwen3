"""集中装配五个 Agent、确定性工具、服务和仓库。"""

from __future__ import annotations

from multi_agent_asr.agents import (
    ASROrchestrator,
    AudioPreparationAgent,
    ContextSelectionAgent,
    RecognitionAgent,
    SupervisorAgent,
    TranscriptReviewAgent,
)
from multi_agent_asr.config import Settings
from multi_agent_asr.memory import ContextBuilder, SqliteMemoryRepository
from multi_agent_asr.observability import SqliteRunRepository
from multi_agent_asr.services import QwenASRService
from multi_agent_asr.tools import (
    HistoryRecorder,
    SceneResolver,
    SoundFileAudioTools,
    SpeakerResolver,
    TerminologyCorrector,
    TranscriptValidator,
)


def build_orchestrator(settings: Settings) -> tuple[ASROrchestrator, QwenASRService]:
    """根据配置装配服务、工具、Agent 和工作流控制器。

    Args:
        settings: 已解析的应用和模型配置。

    Returns:
        完成依赖注入的 Orchestrator，以及供健康检查读取状态的 Qwen 服务。
    """
    settings.ensure_runtime_directories()
    repository = SqliteMemoryRepository(settings.database_path)
    run_repository = SqliteRunRepository(settings.database_path)
    # RecognitionAgent 和健康检查持有同一服务实例，进程内只会懒加载
    # 一份模型权重。
    qwen_service = QwenASRService(settings)

    context_selection_agent = ContextSelectionAgent(
        repository=repository,
        context_builder=ContextBuilder(settings.max_context_chars),
        speaker_resolver=SpeakerResolver(),
        scene_resolver=SceneResolver(),
        recent_limit=settings.recent_utterance_limit,
        min_speaker_confidence=settings.min_speaker_confidence,
        max_steps=settings.context_agent_max_steps,
    )
    orchestrator = ASROrchestrator(
        supervisor_agent=SupervisorAgent(settings.supervisor_max_steps),
        audio_preparation_agent=AudioPreparationAgent(
            tools=SoundFileAudioTools(),
            processed_root=settings.data_root / "processed",
            max_steps=settings.audio_agent_max_steps,
        ),
        context_selection_agent=context_selection_agent,
        recognition_agent=RecognitionAgent(qwen_service, TerminologyCorrector()),
        transcript_review_agent=TranscriptReviewAgent(TranscriptValidator()),
        history_recorder=HistoryRecorder(repository),
        checkpoint_database_path=settings.checkpoint_database_path,
        run_repository=run_repository,
        max_retries=settings.max_asr_retries,
    )
    return orchestrator, qwen_service
