from __future__ import annotations

from typing import Literal

from .state import ASRGraphState


def route_after_verification(state: ASRGraphState) -> Literal["retry", "finalize"]:
    verification = state["verification"]
    if verification.verified or state.get("retry_count", 0) >= state.get("max_retries", 0):
        return "finalize"
    return "retry"
