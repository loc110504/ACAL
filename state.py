
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, TypedDict, Any


@dataclass
class Argument:
    """Represents an argument with its content, type, and validity score"""
    content: str
    argument_type: str  # "support", "attack"
    validity_score: Optional[float] = None
    parent_option: Optional[str] = None
    supporting_docs: List[Dict[str, Any]] = field(default_factory=list)
    agent_role: Optional[str] = None
    agent_name: Optional[str] = None
	
	# metadata: Optional[Dict[str, Any]] = None

class GraphState(TypedDict, total=False):
    """General state schema for legal agentic workflows"""
    # Core task/case data
    task_name: str  # hearsay / learned_hands_courts
    task_info: str  # main input 
    options: List[str] # yes/no options
    arguments: List[Argument]
    validated_arguments: List[Argument]
    final_answer: Optional[str]
    # task_metadata: Optional[Dict[str, Any]]  # task-specific extra info   
    
    # Human interaction and feedback
    human_feedback: Optional[str]
    current_step: str
    human_review_complete: bool
    user_action: Optional[str]

    # RAG and document retrieval
    retrieved_documents: List[Dict[str, Any]]
    search_queries: List[str]
    rag_context: str
    adaptive_retrieval_summary: Optional[Dict[str, Any]]
    document_references: List[Dict[str, Any]]
    cited_documents: Set[int]

    # Multi-agent legal team management
    selected_support_agents: List[Dict[str, Any]]
    selected_attack_agents: List[Dict[str, Any]]
    agent_arguments_tracking: Dict[str, List[Dict[str, Any]]]

    # Streaming and progress tracking
    enable_streaming: bool
    options_generation_progress: Optional[str]
    argument_generation_progress: Optional[str]
    current_argument_stream: Optional[str]
    validation_progress: Optional[str]
    current_validation_stream: Optional[str]
    streaming_chunk: Optional[str]
    partial_response: Optional[str]
    rag_progress: Optional[str] 
    # Scheduling (optional)
    scheduling_query: Optional[Dict[str, Any]]
    scheduling_slots: Optional[Dict[str, List[Dict[str, Any]]]]
    scheduling_summary: Optional[str]   