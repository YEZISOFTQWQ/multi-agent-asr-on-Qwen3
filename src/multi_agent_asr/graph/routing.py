"""定义 Supervisor 决策到 LangGraph 节点的路由。"""

from __future__ import annotations

from multi_agent_asr.schemas import SupervisorAction

from .state import ASRGraphState


def route_supervisor(state: ASRGraphState) -> SupervisorAction:
    """返回 Supervisor 已选择的下一动作。"""
    return state["supervisor_decision"].action
