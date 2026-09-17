from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from multi_agent_asr import __version__
from multi_agent_asr.bootstrap import build_orchestrator
from multi_agent_asr.config import get_settings
from multi_agent_asr.schemas import ASRResult, NodeRunRecord, SpeakerProfile, TranscriptionInput


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    orchestrator, qwen_service = build_orchestrator(settings)
    await orchestrator.initialize()
    app.state.settings = settings
    app.state.orchestrator = orchestrator
    app.state.qwen_service = qwen_service
    try:
        yield
    finally:
        await orchestrator.close()


app = FastAPI(
    title="Multi-Agent ASR",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    qwen_service = request.app.state.qwen_service
    return {
        "status": "ok",
        "version": __version__,
        "environment": settings.environment,
        "model": settings.asr_model_path,
        "model_loaded": qwen_service.is_loaded,
        "device_map": settings.device_map,
        "orchestration": "langgraph",
    }


@app.post("/v1/transcriptions", response_model=ASRResult)
async def transcribe(payload: TranscriptionInput, request: Request) -> ASRResult:
    try:
        return await request.app.state.orchestrator.transcribe(payload)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/runs/{run_id}", response_model=list[NodeRunRecord])
async def get_run(run_id: str, request: Request) -> list[NodeRunRecord]:
    records = await request.app.state.orchestrator.list_node_runs(run_id)
    if not records:
        raise HTTPException(status_code=404, detail="Run not found")
    return records


@app.get("/v1/profiles/{speaker_id}", response_model=SpeakerProfile)
async def get_profile(speaker_id: str, request: Request) -> SpeakerProfile:
    profile = await request.app.state.orchestrator.memory_agent.get_profile(speaker_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Speaker profile not found")
    return profile


@app.put("/v1/profiles/{speaker_id}", response_model=SpeakerProfile)
async def put_profile(
    speaker_id: str,
    profile: SpeakerProfile,
    request: Request,
) -> SpeakerProfile:
    if speaker_id != profile.speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id does not match request path")
    return await request.app.state.orchestrator.memory_agent.upsert_profile(profile)
