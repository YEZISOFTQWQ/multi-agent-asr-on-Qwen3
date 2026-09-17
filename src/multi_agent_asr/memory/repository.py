"""使用 SQLite 持久化说话人画像和会话转写。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from multi_agent_asr.schemas import ASRResult, SpeakerProfile


class SqliteMemoryRepository:
    """持久化说话人画像和按会话组织的转写历史。"""

    def __init__(self, database_path: Path) -> None:
        """保存业务记忆数据库路径。"""
        self.database_path = database_path

    async def initialize(self) -> None:
        """在线程池中创建或迁移记忆表。"""
        await asyncio.to_thread(self._initialize_sync)

    def _connect(self) -> sqlite3.Connection:
        """创建启用名称列访问的短生命周期 SQLite 连接。"""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_sync(self) -> None:
        """同步创建画像、转写和查询索引。"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    speaker_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    accent TEXT,
                    accent_confidence REAL NOT NULL DEFAULT 0,
                    frequent_terms_json TEXT NOT NULL DEFAULT '[]',
                    corrections_json TEXT NOT NULL DEFAULT '{}',
                    environments_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS utterances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    speaker_id TEXT,
                    text TEXT NOT NULL,
                    language TEXT,
                    scene TEXT,
                    verified INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_utterances_session_id
                ON utterances(session_id, id DESC);
                """
            )

    async def get_profile(self, speaker_id: str | None) -> SpeakerProfile | None:
        """异步读取说话人画像。"""
        if not speaker_id:
            return None
        return await asyncio.to_thread(self._get_profile_sync, speaker_id)

    def _get_profile_sync(self, speaker_id: str) -> SpeakerProfile | None:
        """读取一行画像并还原 JSON 字段。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM speaker_profiles WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        if row is None:
            return None
        return SpeakerProfile(
            speaker_id=row["speaker_id"],
            display_name=row["display_name"],
            accent=row["accent"],
            accent_confidence=row["accent_confidence"],
            frequent_terms=json.loads(row["frequent_terms_json"]),
            corrections=json.loads(row["corrections_json"]),
            environments=json.loads(row["environments_json"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def upsert_profile(self, profile: SpeakerProfile) -> SpeakerProfile:
        """异步新增或更新说话人画像。"""
        return await asyncio.to_thread(self._upsert_profile_sync, profile)

    def _upsert_profile_sync(self, profile: SpeakerProfile) -> SpeakerProfile:
        """在一个事务中写入完整画像快照。"""
        updated_at = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO speaker_profiles (
                    speaker_id, display_name, accent, accent_confidence,
                    frequent_terms_json, corrections_json, environments_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(speaker_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    accent = excluded.accent,
                    accent_confidence = excluded.accent_confidence,
                    frequent_terms_json = excluded.frequent_terms_json,
                    corrections_json = excluded.corrections_json,
                    environments_json = excluded.environments_json,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.speaker_id,
                    profile.display_name,
                    profile.accent,
                    profile.accent_confidence,
                    json.dumps(profile.frequent_terms, ensure_ascii=False),
                    json.dumps(profile.corrections, ensure_ascii=False),
                    json.dumps(profile.environments, ensure_ascii=False),
                    updated_at.isoformat(),
                ),
            )
        return profile.model_copy(update={"updated_at": updated_at})

    async def recent_utterances(self, session_id: str, limit: int) -> list[str]:
        """返回会话中最近的已验证转写，顺序从旧到新。"""
        return await asyncio.to_thread(self._recent_utterances_sync, session_id, limit)

    def _recent_utterances_sync(self, session_id: str, limit: int) -> list[str]:
        """查询最近可信文本并恢复自然对话顺序。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT text FROM utterances
                -- 未通过校验的文本不能成为下一轮识别依据。
                WHERE session_id = ? AND verified = 1
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        # SQL 为高效 LIMIT 使用倒序；返回前恢复自然对话顺序。
        return [row["text"] for row in reversed(rows)]

    async def append_utterance(self, result: ASRResult) -> None:
        """异步追加一次最终转写。"""
        await asyncio.to_thread(self._append_utterance_sync, result)

    def _append_utterance_sync(self, result: ASRResult) -> None:
        """在一个事务中写入最终转写及验证状态。"""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO utterances (
                    session_id, speaker_id, text, language, scene, verified, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.session_id,
                    result.speaker_id,
                    result.text,
                    result.language,
                    result.scene,
                    int(result.verified),
                    datetime.now(UTC).isoformat(),
                ),
            )
