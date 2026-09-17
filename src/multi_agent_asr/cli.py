from __future__ import annotations

import argparse
import asyncio
import json

import uvicorn

from multi_agent_asr.bootstrap import build_orchestrator
from multi_agent_asr.config import get_settings
from multi_agent_asr.schemas import TranscriptionInput


def _parser() -> argparse.ArgumentParser:
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
    return parser


async def _initialize() -> None:
    orchestrator, _ = build_orchestrator(get_settings())
    try:
        await orchestrator.initialize()
    finally:
        await orchestrator.close()


async def _transcribe(args: argparse.Namespace) -> None:
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


def main() -> None:
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


if __name__ == "__main__":
    main()
