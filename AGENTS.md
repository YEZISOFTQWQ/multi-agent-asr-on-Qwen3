# Development instructions

- Keep Qwen3-ASR model code in `/home/jiangsongbo/qwen3-asr`; do not copy it into this repository.
- Agents exchange typed objects from `schemas/models.py`.
- New model integrations belong in `services/`; orchestration decisions belong in `agents/`.
- Do not commit audio, model weights, checkpoints, databases, logs, secrets, or generated transcripts.
- Keep model loading lazy so unit tests and API health checks do not require a GPU.
- Context can resolve ambiguity but must never be treated as stronger evidence than audio.
- Add tests for memory migration, context construction, and orchestration decisions.
