"""声明并编译 Multi-Agent ASR 的 LangGraph 拓扑。"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .nodes import ASRGraphNodes
from .routing import route_after_verification
from .state import ASRGraphState


def build_asr_workflow(
    nodes: ASRGraphNodes,
    checkpointer: BaseCheckpointSaver,
):
    """声明并行分析、条件重试和结果持久化的状态图。"""
    builder = StateGraph(ASRGraphState)
    builder.add_node("audio", nodes.audio)
    builder.add_node("speaker", nodes.speaker)
    builder.add_node("scene", nodes.scene)
    builder.add_node("context", nodes.context)
    builder.add_node("asr", nodes.asr)
    builder.add_node("terminology", nodes.terminology)
    builder.add_node("verify", nodes.verify)
    builder.add_node("retry", nodes.retry)
    builder.add_node("finalize", nodes.finalize)
    builder.add_node("persist", nodes.persist)

    # 三个只读分析节点从 START 并行启动；列表边构成全部完成后的汇合屏障。
    builder.add_edge(START, "audio")
    builder.add_edge(START, "speaker")
    builder.add_edge(START, "scene")
    builder.add_edge(["audio", "speaker", "scene"], "context")
    builder.add_edge("context", "asr")
    builder.add_edge("asr", "terminology")
    builder.add_edge("terminology", "verify")
    # 校验结果是唯一控制循环的条件，retry 节点负责递增有界计数。
    builder.add_conditional_edges(
        "verify",
        route_after_verification,
        {"retry": "retry", "finalize": "finalize"},
    )
    builder.add_edge("retry", "asr")
    builder.add_edge("finalize", "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=checkpointer, name="multi-agent-asr")
