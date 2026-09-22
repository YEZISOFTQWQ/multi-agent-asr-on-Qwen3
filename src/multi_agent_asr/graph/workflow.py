"""声明由 Supervisor 驱动的五 Agent LangGraph 拓扑。"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .nodes import ASRGraphNodes
from .routing import route_supervisor
from .state import ASRGraphState


def build_asr_workflow(nodes: ASRGraphNodes, checkpointer: BaseCheckpointSaver):
    """编译 Agent 循环、反馈路由和确定性收尾步骤。

    Args:
        nodes: 已完成依赖注入的图节点适配器。
        checkpointer: 保存每一步共享状态的 LangGraph Checkpoint 实现。

    Returns:
        可异步调用的已编译 LangGraph。
    """
    builder = StateGraph(ASRGraphState)
    builder.add_node("supervisor", nodes.supervisor)
    builder.add_node("audio_preparation", nodes.audio_preparation)
    builder.add_node("context_selection", nodes.context_selection)
    builder.add_node("recognition", nodes.recognition)
    builder.add_node("transcript_review", nodes.transcript_review)
    builder.add_node("finalize", nodes.finalize)
    builder.add_node("persist", nodes.persist)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "audio_preparation": "audio_preparation",
            "context_selection": "context_selection",
            "recognition": "recognition",
            "transcript_review": "transcript_review",
            "finalize": "finalize",
        },
    )
    # 每个 Agent 完成一次动作后都回到 Supervisor 重新观察全局状态。
    # 这组边同时承载 Review -> Supervisor -> Recognition 的有界反馈闭环。
    builder.add_edge("audio_preparation", "supervisor")
    builder.add_edge("context_selection", "supervisor")
    builder.add_edge("recognition", "supervisor")
    builder.add_edge("transcript_review", "supervisor")
    builder.add_edge("finalize", "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=checkpointer, name="supervisor-multi-agent-asr")
