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
    """Persist one auditable record for every LangGraph node execution."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_sync(self) -> None:
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
        return await asyncio.to_thread(self._list_node_runs_sync, run_id)

    def _list_node_runs_sync(self, run_id: str) -> list[NodeRunRecord]:
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
        node_run_id = await self.start_node(
            run_id=run_id,
            thread_id=thread_id,
            node_name=node_name,
            attempt=attempt,
            details=details,
        )
        started = perf_counter()
        try:
            yield
        except BaseException as error:
            await self.fail_node(node_run_id, (perf_counter() - started) * 1000, error)
            raise
        else:
            await self.finish_node(node_run_id, (perf_counter() - started) * 1000)
