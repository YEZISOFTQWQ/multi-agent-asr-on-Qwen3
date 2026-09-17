"""定义 LangGraph 的条件分支规则。"""

from __future__ import annotations

from typing import Literal

from .state import ASRGraphState


def route_after_verification(state: ASRGraphState) -> Literal["retry", "finalize"]:
    """在校验通过或耗尽额度时结束，否则进入重试节点。"""
    verification = state["verification"]
    if verification.verified or state.get("retry_count", 0) >= state.get("max_retries", 0):
        return "finalize"
    return "retry"
