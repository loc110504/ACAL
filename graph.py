from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from node_clash_resolution import (
    argument_validator,
    multi_agent_argument_generator,
    # overall_plan_generator,
    human_review,
    rag_retrieval,
    route_after_human_review,
    final_answer_generator,
)
from state import GraphState


def create_care_plan_graph():
    """Create and configure the LangGraph workflow"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("rag_retrieval", rag_retrieval)
    # workflow.add_node("overall_plan_generation", overall_plan_generator)
    workflow.add_node("argument_generation", multi_agent_argument_generator)
    workflow.add_node("human_review", human_review)
    workflow.add_node("argument_validation", argument_validator)
    workflow.add_node("final_answer_generation", final_answer_generator)
    
    # Add edges
    workflow.set_entry_point("rag_retrieval")
    workflow.add_edge("rag_retrieval", "overall_plan_generation")
    workflow.add_edge("overall_plan_generation", "argument_generation")
    workflow.add_edge("argument_generation", "human_review")

    # Conditional edge for human review
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"human_review": "human_review", "argument_validation": "argument_validation"},
    )

    workflow.add_edge("argument_validation", "final_answer_generation")
    workflow.add_edge("scheduling", END)

    # Compile with memory for checkpointing
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
