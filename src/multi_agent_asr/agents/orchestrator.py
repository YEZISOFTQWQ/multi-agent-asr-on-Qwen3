"""管理 LangGraph 工作流的生命周期并提供统一转写入口。"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_agent_asr.agents.audio_agent import AudioAgent
from multi_agent_asr.agents.memory_agent import MemoryAgent
from multi_agent_asr.agents.profile_update_agent import ProfileUpdateAgent
from multi_agent_asr.agents.qwen_asr_agent import QwenASRAgent
from multi_agent_asr.agents.scene_agent import SceneAgent
from multi_agent_asr.agents.speaker_agent import SpeakerAgent
from multi_agent_asr.agents.terminology_agent import TerminologyAgent
from multi_agent_asr.agents.verifier_agent import VerifierAgent
from multi_agent_asr.graph import ASRGraphNodes, build_asr_workflow
from multi_agent_asr.observability import SqliteRunRepository
from multi_agent_asr.schemas import ASRResult, NodeRunRecord, TranscriptionInput


class ASROrchestrator:
    """管理持久化 LangGraph，并保持稳定的转写调用接口。"""

    def __init__(
        self,
        *,
        audio_agent: AudioAgent,
        speaker_agent: SpeakerAgent,
        scene_agent: SceneAgent,
        memory_agent: MemoryAgent,
        asr_agent: QwenASRAgent,
        terminology_agent: TerminologyAgent,
        verifier_agent: VerifierAgent,
        profile_update_agent: ProfileUpdateAgent,
        checkpoint_database_path: Path | None = None,
        run_repository: SqliteRunRepository | None = None,
        max_retries: int = 1,
    ) -> None:
        """注入所有 Agent，并配置 Checkpoint、运行记录和重试上限。"""
        self.audio_agent = audio_agent
        self.speaker_agent = speaker_agent
        self.scene_agent = scene_agent
        self.memory_agent = memory_agent
        self.asr_agent = asr_agent
        self.terminology_agent = terminology_agent
        self.verifier_agent = verifier_agent
        self.profile_update_agent = profile_update_agent
        self.max_retries = max(0, max_retries)

        # 直接构造编排器时，让 Checkpoint 默认落在记忆库的同级目录。
        memory_database_path = memory_agent.repository.database_path
        self.checkpoint_database_path = checkpoint_database_path or memory_database_path.with_name(
            "checkpoints.sqlite3"
        )
        self.run_repository = run_repository or SqliteRunRepository(memory_database_path)

        self._initialization_lock = asyncio.Lock()
        self._checkpointer_context: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: Any | None = None

    async def initialize(self) -> None:
        """初始化仓库、打开 SQLite Checkpoint 并编译状态图。"""
        await self.memory_agent.initialize()
        await self.run_repository.initialize()
        if self._graph is not None:
            return

        async with self._initialization_lock:
            if self._graph is not None:
                return
            self.checkpoint_database_path.parent.mkdir(parents=True, exist_ok=True)
            # Saver 自身是异步上下文管理器，必须保持到应用关闭才能安全复用连接。
            context = AsyncSqliteSaver.from_conn_string(str(self.checkpoint_database_path))
            checkpointer = await context.__aenter__()
            try:
                await checkpointer.setup()
                nodes = ASRGraphNodes(
                    audio_agent=self.audio_agent,
                    speaker_agent=self.speaker_agent,
                    scene_agent=self.scene_agent,
                    memory_agent=self.memory_agent,
                    asr_agent=self.asr_agent,
                    terminology_agent=self.terminology_agent,
                    verifier_agent=self.verifier_agent,
                    profile_update_agent=self.profile_update_agent,
                    run_repository=self.run_repository,
                )
                graph = build_asr_workflow(nodes, checkpointer)
            except BaseException:
                # 初始化中途失败也要关闭已经打开的 aiosqlite 连接。
                await context.__aexit__(None, None, None)
                raise
            self._checkpointer_context = context
            self._checkpointer = checkpointer
            self._graph = graph

    async def close(self) -> None:
        """幂等关闭 Checkpoint 上下文及其异步 SQLite 连接。"""
        async with self._initialization_lock:
            context = self._checkpointer_context
            self._graph = None
            self._checkpointer = None
            self._checkpointer_context = None
            if context is not None:
                await context.__aexit__(None, None, None)

    async def transcribe(self, request: TranscriptionInput) -> ASRResult:
        """为请求创建独立图线程，执行工作流并返回最终结果。"""
        if self._graph is None:
            raise RuntimeError("ASROrchestrator.initialize() must be called before transcribe()")

        run_id = uuid4().hex
        # 同一 session 的每次请求也使用独立 thread，避免续接旧 Checkpoint。
        thread_id = f"{request.session_id}:{run_id}"
        final_state = await self._graph.ainvoke(
            {
                "request": request,
                "run_id": run_id,
                "thread_id": thread_id,
                "retry_count": 0,
                "max_retries": self.max_retries,
                "retry_reason": None,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return ASRResult.model_validate(final_state["result"])

    async def list_node_runs(self, run_id: str) -> list[NodeRunRecord]:
        """返回指定运行的节点级审计记录。"""
        return await self.run_repository.list_node_runs(run_id)
