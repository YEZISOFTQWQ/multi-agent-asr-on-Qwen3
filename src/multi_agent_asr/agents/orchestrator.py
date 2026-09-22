"""管理 Supervisor 多 Agent 工作流的生命周期和统一入口。"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_agent_asr.agents.audio_preparation_agent import AudioPreparationAgent
from multi_agent_asr.agents.context_selection_agent import ContextSelectionAgent
from multi_agent_asr.agents.recognition_agent import RecognitionAgent
from multi_agent_asr.agents.supervisor_agent import SupervisorAgent
from multi_agent_asr.agents.transcript_review_agent import TranscriptReviewAgent
from multi_agent_asr.graph import ASRGraphNodes, build_asr_workflow
from multi_agent_asr.observability import SqliteRunRepository
from multi_agent_asr.schemas import ASRResult, NodeRunRecord, TranscriptionInput
from multi_agent_asr.tools import HistoryRecorder


class ASROrchestrator:
    """持有工作流资源，并把每次转写交给 Supervisor 驱动。

    Orchestrator 负责仓库、Checkpoint 和已编译 LangGraph 的生命周期，
    并为每个请求创建隔离的运行标识。路由决策由 SupervisorAgent 完成，
    因此该类是应用服务和资源控制器。
    """

    def __init__(
        self,
        *,
        supervisor_agent: SupervisorAgent,
        audio_preparation_agent: AudioPreparationAgent,
        context_selection_agent: ContextSelectionAgent,
        recognition_agent: RecognitionAgent,
        transcript_review_agent: TranscriptReviewAgent,
        history_recorder: HistoryRecorder,
        checkpoint_database_path: Path | None = None,
        run_repository: SqliteRunRepository | None = None,
        max_retries: int = 1,
    ) -> None:
        self.supervisor_agent = supervisor_agent
        self.audio_preparation_agent = audio_preparation_agent
        self.context_selection_agent = context_selection_agent
        self.recognition_agent = recognition_agent
        self.transcript_review_agent = transcript_review_agent
        self.history_recorder = history_recorder
        self.max_retries = max(0, max_retries)

        # 默认从业务数据库旁派生 Checkpoint 路径，使最小配置即可运行，
        # 同时仍允许生产环境显式分库。
        memory_database_path = context_selection_agent.repository.database_path
        self.checkpoint_database_path = checkpoint_database_path or memory_database_path.with_name(
            "checkpoints.sqlite3"
        )
        self.run_repository = run_repository or SqliteRunRepository(memory_database_path)
        self._initialization_lock = asyncio.Lock()
        self._checkpointer_context: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: Any | None = None

    async def initialize(self) -> None:
        """幂等初始化仓库、Checkpoint 和 Supervisor 状态图。"""
        await self.context_selection_agent.initialize()
        await self.run_repository.initialize()
        if self._graph is not None:
            return

        # FastAPI 生命周期和直接调用可能同时触发初始化。锁内再次检查，
        # 保证进程内只持有一个 Checkpoint 上下文和一份编译图。
        async with self._initialization_lock:
            if self._graph is not None:
                return
            self.checkpoint_database_path.parent.mkdir(parents=True, exist_ok=True)
            context = AsyncSqliteSaver.from_conn_string(str(self.checkpoint_database_path))
            checkpointer = await context.__aenter__()
            try:
                await checkpointer.setup()
                nodes = ASRGraphNodes(
                    supervisor_agent=self.supervisor_agent,
                    audio_preparation_agent=self.audio_preparation_agent,
                    context_selection_agent=self.context_selection_agent,
                    recognition_agent=self.recognition_agent,
                    transcript_review_agent=self.transcript_review_agent,
                    history_recorder=self.history_recorder,
                    run_repository=self.run_repository,
                )
                graph = build_asr_workflow(nodes, checkpointer)
            except BaseException:
                # __aenter__ 成功后的任何初始化失败都要配对退出，避免遗留
                # SQLite 连接或后台线程。
                await context.__aexit__(None, None, None)
                raise
            self._checkpointer_context = context
            self._checkpointer = checkpointer
            self._graph = graph

    async def close(self) -> None:
        """幂等关闭异步 Checkpoint 连接。"""
        async with self._initialization_lock:
            context = self._checkpointer_context
            self._graph = None
            self._checkpointer = None
            self._checkpointer_context = None
            if context is not None:
                await context.__aexit__(None, None, None)

    async def transcribe(self, request: TranscriptionInput) -> ASRResult:
        """创建独立运行并执行 Supervisor 驱动的 Agent 循环。

        Args:
            request: 已通过 Pydantic 校验的转写请求。

        Returns:
            图完成审核和持久化后的最终转写结果。

        Raises:
            RuntimeError: Orchestrator 尚未完成初始化。
        """
        if self._graph is None:
            raise RuntimeError("ASROrchestrator.initialize() must be called before transcribe()")

        # run_id 标识一次业务运行；thread_id 同时包含 session_id，使
        # Checkpoint 可按会话定位，又不会让两次请求共享可变状态。
        run_id = uuid4().hex
        thread_id = f"{request.session_id}:{run_id}"
        final_state = await self._graph.ainvoke(
            {
                "request": request,
                "run_id": run_id,
                "thread_id": thread_id,
                "supervisor_step": 0,
                "supervisor_history": [],
                "audio_preparation": None,
                "audio": None,
                "context_selection": None,
                "context": None,
                "recognition": None,
                "candidate": None,
                "review": None,
                "verification": None,
                "retry_count": 0,
                "max_retries": self.max_retries,
                "retry_feedback": None,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return ASRResult.model_validate(final_state["result"])

    async def list_node_runs(self, run_id: str) -> list[NodeRunRecord]:
        """返回指定运行的节点级审计记录。"""
        return await self.run_repository.list_node_runs(run_id)
