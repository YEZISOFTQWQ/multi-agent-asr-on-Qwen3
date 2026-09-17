"""使用 SQLite 记录每个 LangGraph 节点的执行情况。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from multi_agent_asr.schemas import NodeRunRecord


class SqliteRunRepository:
    """为每次 LangGraph 节点执行保存一条可审计记录。"""

    def __init__(self, database_path: Path) -> None:
        """保存节点运行记录所在的数据库路径。"""
        self.database_path = database_path

    async def initialize(self) -> None:
        """在线程池中创建节点记录表和索引。"""
        await asyncio.to_thread(self._initialize_sync)

    def _connect(self) -> sqlite3.Connection:
        """创建允许并发任务等待写锁的 SQLite 连接。"""
        # 并行图节点可能同时落库，timeout 允许短暂等待 SQLite 写锁。
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_sync(self) -> None:
        """同步创建节点记录表。"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS node_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL,
                    attempt INTEGER NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_node_runs_run_id
                ON node_runs(run_id, id);
                """
            )

    async def start_node(
        self,
        *,
        run_id: str,
        thread_id: str,
        node_name: str,
        attempt: int,
        details: dict[str, object] | None = None,
    ) -> int:
        """插入 running 状态并返回节点记录 ID。"""
        return await asyncio.to_thread(
            self._start_node_sync,
            run_id,
            thread_id,
            node_name,
            attempt,
            details or {},
        )

    def _start_node_sync(
        self,
        run_id: str,
        thread_id: str,
        node_name: str,
        attempt: int,
        details: dict[str, object],
    ) -> int:
        """同步持久化节点开始时间和静态明细。"""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO node_runs (
                    run_id, thread_id, node_name, status, started_at, attempt, details_json
                ) VALUES (?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    run_id,
                    thread_id,
                    node_name,
                    datetime.now(UTC).isoformat(),
                    attempt,
                    json.dumps(details, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    async def finish_node(self, node_run_id: int, duration_ms: float) -> None:
        """把节点记录更新为成功并保存耗时。"""
        await asyncio.to_thread(
            self._finish_node_sync,
            node_run_id,
            "succeeded",
            duration_ms,
            None,
        )

    async def fail_node(
        self,
        node_run_id: int,
        duration_ms: float,
        error: BaseException,
    ) -> None:
        """把节点记录更新为失败并保存异常摘要。"""
        await asyncio.to_thread(
            self._finish_node_sync,
            node_run_id,
            "failed",
            duration_ms,
            f"{type(error).__name__}: {error}",
        )

    def _finish_node_sync(
        self,
        node_run_id: int,
        status: str,
        duration_ms: float,
        error: str | None,
    ) -> None:
        """同步完成节点记录的终态更新。"""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE node_runs
                SET status = ?, finished_at = ?, duration_ms = ?, error = ?
                WHERE id = ?
                """,
                (
                    status,
                    datetime.now(UTC).isoformat(),
                    duration_ms,
                    error,
                    node_run_id,
                ),
            )

    async def list_node_runs(self, run_id: str) -> list[NodeRunRecord]:
        """按写入顺序返回一次运行的所有节点记录。"""
        return await asyncio.to_thread(self._list_node_runs_sync, run_id)

    def _list_node_runs_sync(self, run_id: str) -> list[NodeRunRecord]:
        """查询记录并把 SQLite 行转换为强类型模型。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM node_runs WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            NodeRunRecord(
                id=row["id"],
                run_id=row["run_id"],
                thread_id=row["thread_id"],
                node_name=row["node_name"],
                status=row["status"],
                started_at=datetime.fromisoformat(row["started_at"]),
                finished_at=(
                    datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
                ),
                duration_ms=row["duration_ms"],
                attempt=row["attempt"],
                details=json.loads(row["details_json"]),
                error=row["error"],
            )
            for row in rows
        ]

    @asynccontextmanager
    async def track(
        self,
        *,
        run_id: str,
        thread_id: str,
        node_name: str,
        attempt: int,
        details: dict[str, object] | None = None,
    ) -> AsyncIterator[None]:
        """在异步上下文中自动记录节点成功、失败和耗时。"""
        node_run_id = await self.start_node(
            run_id=run_id,
            thread_id=thread_id,
            node_name=node_name,
            attempt=attempt,
            details=details,
        )
        # perf_counter 不受系统时钟校准影响，适合计算节点耗时。
        started = perf_counter()
        try:
            yield
        except BaseException as error:
            # 取消等控制流异常也要留下失败记录，随后原样抛给 LangGraph。
            await self.fail_node(node_run_id, (perf_counter() - started) * 1000, error)
            raise
        else:
            await self.finish_node(node_run_id, (perf_counter() - started) * 1000)
