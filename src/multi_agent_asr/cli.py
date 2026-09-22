"""Multi-Agent ASR 的命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json

import uvicorn

from multi_agent_asr.bootstrap import build_orchestrator
from multi_agent_asr.config import get_settings
from multi_agent_asr.schemas import TranscriptionInput


def _parser() -> argparse.ArgumentParser:
    """构建初始化、服务、转写和 Agent 日志子命令。"""
    parser = argparse.ArgumentParser(description="Multi-Agent ASR command line")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init-db", help="Create runtime directories and initialize SQLite")
    commands.add_parser("serve", help="Start the FastAPI server")

    transcribe = commands.add_parser("transcribe", help="Transcribe one local audio file")
    transcribe.add_argument("audio_path")
    transcribe.add_argument("--session-id", default="default")
    transcribe.add_argument("--speaker")
    transcribe.add_argument("--scene")
    transcribe.add_argument("--language")
    transcribe.add_argument("--context")
    transcribe.add_argument("--timestamps", action="store_true")

    agent_log = commands.add_parser(
        "agent-log",
        help="Show structured agent logs for one run",
    )
    agent_log.add_argument("run_id", help="run_id returned by transcribe")
    return parser


async def _initialize() -> None:
    """初始化数据库和 Checkpoint 表后释放资源。"""
    orchestrator, _ = build_orchestrator(get_settings())
    try:
        await orchestrator.initialize()
    finally:
        await orchestrator.close()


async def _transcribe(args: argparse.Namespace) -> None:
    """执行一次 CLI 转写并以 JSON 输出结构化结果。"""
    orchestrator, _ = build_orchestrator(get_settings())
    try:
        await orchestrator.initialize()
        result = await orchestrator.transcribe(
            TranscriptionInput(
                audio_path=args.audio_path,
                session_id=args.session_id,
                speaker_hint=args.speaker,
                scene_hint=args.scene,
                language=args.language,
                explicit_context=args.context,
                return_time_stamps=args.timestamps,
            )
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        await orchestrator.close()


async def _agent_log(args: argparse.Namespace) -> None:
    """按时间顺序输出一次运行中所有 Agent 的结构化日志。"""
    orchestrator, _ = build_orchestrator(get_settings())
    try:
        await orchestrator.initialize()
        records = await orchestrator.list_node_runs(args.run_id)
        agent_records = [record for record in records if record.details.get("agent")]
        if not agent_records:
            raise SystemExit(f"No agent logs found for run_id: {args.run_id}")
        print(
            json.dumps(
                [record.model_dump(mode="json") for record in agent_records],
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        await orchestrator.close()


def main() -> None:
    """解析命令行参数并分派到对应子命令。"""
    args = _parser().parse_args()
    settings = get_settings()
    if args.command == "init-db":
        asyncio.run(_initialize())
    elif args.command == "serve":
        uvicorn.run(
            "multi_agent_asr.api.app:app",
            host=settings.host,
            port=settings.port,
            reload=settings.environment == "development",
        )
    elif args.command == "transcribe":
        asyncio.run(_transcribe(args))
    elif args.command == "agent-log":
        asyncio.run(_agent_log(args))


if __name__ == "__main__":
    main()
