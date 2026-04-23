from typing import Literal
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from state import SupportState
from nodes import classify_intent, technical_node, billing_node, general_node, summarize_node

def route_by_intent(state: SupportState) -> Literal["technical", "billing", "general"]:
    return state.get("intent", "general")

def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify", classify_intent)
    graph.add_node("technical", technical_node)
    graph.add_node("billing", billing_node)
    graph.add_node("general", general_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", route_by_intent, {
        "technical": "technical",
        "billing": "billing",
        "general": "general",
    })

    graph.add_edge("technical", "summarize")
    graph.add_edge("billing", "summarize")
    graph.add_edge("general", "summarize")
    graph.set_finish_point("summarize")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)