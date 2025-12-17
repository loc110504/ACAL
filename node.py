from state import Argument, GraphState
from llm_caller import call_llm, call_llm_stream
import re
import json

from legal_agents import LEGAL_AGENTS
from test import RAGModule

def rag_retrieval(state: GraphState) -> GraphState:
    print("[STEP] rag_retrieval: Starting document retrieval for task:", state.get('task_name'))
    # Use RAGModule from test.py for vector search
    rag = RAGModule(persist_directory="./chroma_db")
    collection_name = "phq8_medical_docs"
    rag.load_collection(collection_name)

    if state['task_name'] == 'hearsay':
        prompt = f"""
        Based on the following case information, retrieve relevant legal documents that discuss hearsay evidence and its admissibility in court:

        {state['task_info']}

        IMPORTANT: Focus queries on evidentiary rules, exceptions, and case law about hearsay.
        """
    elif state['task_name'] == 'learned_hands_courts':
        prompt = f"""
        Based on the following case information, retrieve relevant legal documents that discuss the Learned Hand formula and its application in negligence cases:

        {state['task_info']}
        IMPORTANT: Focus queries on judicial reasoning, precedent, and relevant statutes.
        """
    else:
        prompt = f"""
        Based on the following case information, retrieve relevant legal documents that are pertinent to the legal issues presented:

        {state['task_info']}
        """
    if state.get("enable_streaming", False):
        queries_text = ""
        for chunk in call_llm_stream(prompt, temperature=0.3, max_tokens=256):
            queries_text += chunk
            state["rag_progress"] = f"Generating search queries:\n{queries_text}"
        queries = queries_text.strip().split("\n")
    else:
        queries = call_llm(prompt, temperature=0.3, max_tokens=256).strip().split("\n")

    state["search_queries"] = queries

    all_documents = []
    seen_docs = set()  # Avoid duplicates
    document_references = []
    doc_id = 1
    for query in queries[:5]:  # Limit to first 5 queries
        if query.strip():
            results = rag.query_rag(query=query.strip(), top_k=3, collection_name=collection_name)
            for doc in results:
                doc_text = doc["text"]
                if doc_text not in seen_docs:
                    doc_with_id = {
                        "doc_id": doc_id,
                        "search_query": query.strip(),
                        "document": doc_text,
                        "similarity_score": doc.get("score", 0.0),
                        "metadata": {"source": doc.get("source", "unknown"), "chunk_id": doc.get("chunk_id", None)}
                    }
                    all_documents.append(doc_with_id)
                    seen_docs.add(doc_text)
                    reference = {
                        "id": doc_id,
                        "content": (
                            doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                        ),
                        "full_content": doc_text,
                        "similarity_score": doc.get("score", 0.0),
                        "metadata": {"source": doc.get("source", "unknown"), "chunk_id": doc.get("chunk_id", None)},
                        "search_query": query.strip(),
                        "used_in": [],
                    }
                    document_references.append(reference)
                    doc_id += 1

    state["retrieved_documents"] = all_documents
    state["document_references"] = document_references
    state["cited_documents"] = set()

    rag_context = f"RELEVANT LEGAL ISSUES DOCUMENTS IN {state['task_name']}:\n\n"
    for doc in all_documents[:10]:  # Limit context
        rag_context += (
            f"[REF-{doc['doc_id']}] (Relevance: {doc['similarity_score']:.2f})\n"
        )
        rag_context += f"{doc['document']}\n\n"

    state["rag_context"] = rag_context
    print(f"Retrieved {len(all_documents)} relevant documents with references")
    return state

def overall_options(state: GraphState) -> GraphState:
    print("[STEP] overall_options: Setting options for task:", state.get('task_name'))
    # Hearsay, Learned Hands, etc. all have Yes/No options
    options = ["Yes", "No"]
    state["options"] = options
    state["current_step"] = "argument_generation"
    return state

def agent_selector(state: GraphState, type: str) -> GraphState:
    print(f"[STEP] agent_selector: Selecting {type} agents for task:", state.get('task_name'))
    # Type = Support / Attack
    available_agents = []

    # All agents are available except Judge

    for role, agent in LEGAL_AGENTS.items():
        available_agents.append({
            "role": role.value,
            "name": agent.name,
            "expertise": agent.expertise_areas
        })
    selection_prompt = f"""
    Based on the following case information, select the most suitable legal agents to provide {type}ing arguments.
    Case Information:
    {state['task_info']}
    AVAILABLE AGENTS:
    {available_agents}
    IMPORTANT: Choose agents whose expertise aligns closely with the legal issues presented and the nature of the arguments needed.
    Return your selection strictly as a valid JSON object. Use double quotes around keys and string values, and do not include any trailing commas. The JSON must have this structure:
    {{
        "selected_agents": [
            {{
                "role": "Private Practice Lawyer",
                "reason": "Why this professional is essential for this patient"
            }},
            {{
                "role": "Corporate / In-house Counsel",
                "reason": "Why this professional is essential for this patient"
            }}
            ...
        ]
    }}
    Provide only the JSON object in your response. Do not include any explanation outside the JSON.
    """

    selection_response = call_llm(selection_prompt, temperature=0.3, max_tokens=512)
    
    # Parse JSON response and extract selected_agents list
    try:
        selection_json = json.loads(selection_response)
        selected_agents = selection_json.get("selected_agents", [])
        if not isinstance(selected_agents, list):
            print(f"[ERROR] 'selected_agents' is not a list. Got: {type(selected_agents)}")
            selected_agents = []
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse agent selection JSON: {e}")
        print(f"Response was: {selection_response[:200]}...")
        selected_agents = []
    except Exception as e:
        print(f"[ERROR] Unexpected error parsing agent selection: {e}")
        selected_agents = []

    if type == "support":
        state["selected_support_agents"] = selected_agents
    elif type == "attack":
        state["selected_attack_agents"] = selected_agents
    
    return state

def multi_agent_argument_generator(state: GraphState) -> GraphState:
    print("[STEP] multi_agent_argument_generator: Generating arguments for options:", state.get('options', []))

    arguments = []
    rag_context = state.get("rag_context", "")
    options = state.get("options", [])

    agent_arguments_tracking = {}

    # Generate support arguments for each option using selected support agents
    for option in options:
        option_arguments = []
        for agent_info in state.get("selected_support_agents", []):
            agent_role = agent_info.get("role", "")
            agent_name = agent_info.get("name", "")
            support_prompt = f"""
            As a legal expert ({agent_role}), provide 1-2 supporting arguments for the following legal option:
            {rag_context}
            Legal Option Being Evaluated in the task {state.get("task_info", "")}:
            {option}
            Focus on aspects most relevant to your expertise.
            Format each argument on a new line starting with 'Support:'
            """
            support_response = call_llm(support_prompt, temperature=0.7, max_tokens=256)
            for line in support_response.split("\n"):
                if line.strip().startswith("Support:"):
                    arg_content = line.replace("Support:", "").strip()
                    if arg_content:
                        tagged_content = f"[{agent_role}] {arg_content}"
                        arg = Argument(
                            content=tagged_content,
                            argument_type="support",
                            parent_option=option,
                        )
                        arg.agent_role = agent_role
                        arg.agent_name = agent_name
                        arguments.append(arg)
                        option_arguments.append(arg)
                        if agent_name not in agent_arguments_tracking:
                            agent_arguments_tracking[agent_name] = []
                        agent_arguments_tracking[agent_name].append({
                            "type": "support",
                            "content": arg_content,
                            "option": option
                        })

        # Generate attack arguments for each option using selected attack agents
        for agent_info in state.get("selected_attack_agents", []):
            agent_role = agent_info.get("role", "")
            agent_name = agent_info.get("name", "")
            attack_prompt = f"""
            As a legal expert ({agent_role}), provide 1-2 challenging arguments for the following legal option:
            {rag_context}
            Legal Option Being Evaluated in the task {state.get("task_info", "")}:
            {option}
            Focus on risks or limitations most relevant to your expertise.
            Format each argument on a new line starting with 'Challenge:' or 'Attack:'
            """
            attack_response = call_llm(attack_prompt, temperature=0.7, max_tokens=256)
            for line in attack_response.split("\n"):
                if line.strip().startswith("Challenge:") or line.strip().startswith("Attack:"):
                    arg_content = line.replace("Challenge:", "").replace("Attack:", "").strip()
                    if arg_content:
                        tagged_content = f"[{agent_role}] {arg_content}"
                        arg = Argument(
                            content=tagged_content,
                            argument_type="attack",
                            parent_option=option,
                        )
                        arg.agent_role = agent_role
                        arg.agent_name = agent_name
                        arguments.append(arg)
                        option_arguments.append(arg)
                        if agent_name not in agent_arguments_tracking:
                            agent_arguments_tracking[agent_name] = []
                        agent_arguments_tracking[agent_name].append({
                            "type": "attack",
                            "content": arg_content,
                            "option": option
                        })

    state["agent_arguments_tracking"] = agent_arguments_tracking
    state["arguments"] = arguments
    state["current_step"] = "human_review"
    if "human_review_complete" not in state:
        state["human_review_complete"] = False
    return state

def human_review(state: GraphState) -> GraphState:
    print("[STEP] human_review: Human review step (placeholder)")
    return state    

def route_after_human_review(state: GraphState) -> str:
    print("[STEP] route_after_human_review: Routing after human review. Complete:", state.get('human_review_complete', False))
    if state.get("human_review_complete", False):
        return "argument_validation"
    else:
        return "human_review"

def argument_validator(state: GraphState) -> GraphState:
    print("[STEP] argument_validator: Validating arguments for task:", state.get('task_name'))
    validated_arguments = []
    rag = RAGModule(persist_directory="./chroma_db")
    arguments_with_additional_evidence = []
    total_args = len(state.get("arguments", []))

    for i, arg in enumerate(state.get("arguments", [])):
        if state.get("enable_streaming", False):
            state["validation_progress"] = f"Validating argument {i+1}/{total_args}"

        # Build prompt based on task
        if state.get("task_name") == "hearsay":
            initial_prompt = f"""
                You are a legal analyst evaluating the validity and relevance of arguments about hearsay evidence.
                Legal Option: {arg.parent_option}
                Argument ({arg.argument_type}): {arg.content}
                Please evaluate this argument based on:
                1. Factual accuracy
                2. Relevance to hearsay rules and exceptions
                3. Practical considerations in court
                4. Evidence-based legal reasoning
                Provide a validity score between 0 and 1, where:
                - 0 = completely invalid/irrelevant
                - 0.5 = moderately valid
                - 1 = highly valid and relevant
                Response format: "Validity Score: X.XX"
                Include a brief explanation."""
        elif state.get("task_name") == "learned_hands_courts":
            initial_prompt = f"""
                You are a legal analyst evaluating the validity and relevance of arguments about the Learned Hand formula in negligence cases.
                Legal Option: {arg.parent_option}
                Argument ({arg.argument_type}): {arg.content}
                Please evaluate this argument based on:
                1. Factual accuracy
                2. Relevance to the Learned Hand formula and judicial reasoning
                3. Practical considerations in negligence law
                4. Evidence-based legal reasoning
                Provide a validity score between 0 and 1, where:
                - 0 = completely invalid/irrelevant
                - 0.5 = moderately valid
                - 1 = highly valid and relevant
                Response format: "Validity Score: X.XX"
                Include a brief explanation."""
        else:
            initial_prompt = f"""
                You are a legal analyst evaluating the validity and relevance of arguments for this legal case.
                Legal Option: {arg.parent_option}
                Argument ({arg.argument_type}): {arg.content}
                Please evaluate this argument based on:
                1. Factual accuracy
                2. Relevance to the legal issues
                3. Practical considerations
                4. Evidence-based legal reasoning
                Provide a validity score between 0 and 1, where:
                - 0 = completely invalid/irrelevant
                - 0.5 = moderately valid
                - 1 = highly valid and relevant
                Response format: "Validity Score: X.XX"
                Include a brief explanation."""

        if state.get("enable_streaming", False):
            initial_response = ""
            for chunk in call_llm_stream(initial_prompt, temperature=0.3, max_tokens=256):
                initial_response += chunk
                state["current_validation_stream"] = initial_response
        else:
            initial_response = call_llm(initial_prompt, temperature=0.3, max_tokens=256)

        initial_validity_score = 0.5
        try:
            if "Validity Score:" in initial_response:
                score_text = initial_response.split("Validity Score:")[1].split()[0]
                initial_validity_score = float(score_text.strip())
                initial_validity_score = max(0, min(1, initial_validity_score))
        except:
            pass

        # If weak, retrieve additional legal evidence
        if initial_validity_score < 0.5:
            print(f"Low validity score ({initial_validity_score:.2f}) for argument. Retrieving additional legal evidence...")
            search_query_prompt = f"""
                Generate a specific legal search query to find evidence about this argument:
                Legal Option: {arg.parent_option}
                Argument Type: {arg.argument_type}
                Argument: {arg.content}
                Create ONE focused search query that would help validate or refute this argument."""
            if state.get("enable_streaming", False):
                search_query = ""
                for chunk in call_llm_stream(search_query_prompt, temperature=0.2, max_tokens=128):
                    search_query += chunk
            else:
                search_query = call_llm(search_query_prompt, temperature=0.2, max_tokens=128).strip()

                additional_docs = rag.query_rag(query=search_query, top_k=3, collection_name="legal_docs")
            doc_refs_used = []
            additional_context = "\nADDITIONAL LEGAL EVIDENCE:\n"
            for idx, doc in enumerate(additional_docs, 1):
                additional_context += f"\n[Evidence {idx}] (Relevance: {doc.get('score', 0.0):.2f})\n"
                additional_context += f"{doc['text'][:500]}...\n"
                if "document_references" in state:
                    for ref in state["document_references"]:
                            if ref["full_content"] == doc["text"]:
                                ref["used_in"].append(f"validation_{arg.content[:30]}...")
                                doc_refs_used.append(ref["id"])
                                break
            arg.supporting_docs = doc_refs_used

            # Re-validate with additional context
            revalidation_prompt = f"""
                You are a legal analyst. Re-evaluate this argument with additional legal evidence.
                Legal Option: {arg.parent_option}
                Argument ({arg.argument_type}): {arg.content}
                Initial Assessment: The argument initially scored {initial_validity_score:.2f} in validity.
                {additional_context}
                Based on the additional evidence above, re-evaluate this argument considering:
                1. Whether the evidence supports or contradicts the argument
                2. The reliability and relevance of the evidence
                3. How this changes the practical validity of the argument
                Provide an updated validity score between 0 and 1.
                Response format: "Updated Validity Score: X.XX"
                Explain how the evidence influenced your assessment."""
            revalidation_response = call_llm(revalidation_prompt, temperature=0.3, max_tokens=512)
            updated_validity_score = initial_validity_score
            try:
                if "Updated Validity Score:" in revalidation_response:
                    score_text = revalidation_response.split("Updated Validity Score:")[1].split()[0]
                    updated_validity_score = float(score_text.strip())
                    updated_validity_score = max(0, min(1, updated_validity_score))
                elif "Validity Score:" in revalidation_response:
                    score_text = revalidation_response.split("Validity Score:")[1].split()[0]
                    updated_validity_score = float(score_text.strip())
                    updated_validity_score = max(0, min(1, updated_validity_score))
            except:
                pass
            arguments_with_additional_evidence.append({
                "argument": arg.content,
                "initial_score": initial_validity_score,
                "updated_score": updated_validity_score,
                "evidence_used": len(additional_docs),
            })
            arg.validity_score = updated_validity_score
            print(f"  Score updated: {initial_validity_score:.2f} → {updated_validity_score:.2f}")
        else:
            arg.validity_score = initial_validity_score
            print(f"Argument validity score: {initial_validity_score:.2f}")
        validated_arguments.append(arg)

    # Add summary of adaptive retrieval to state
    if arguments_with_additional_evidence:
        adaptive_retrieval_summary = {
            "arguments_enhanced": len(arguments_with_additional_evidence),
            "average_score_improvement": sum(
                a["updated_score"] - a["initial_score"] for a in arguments_with_additional_evidence
            ) / len(arguments_with_additional_evidence),
            "details": arguments_with_additional_evidence,
        }
        state["adaptive_retrieval_summary"] = adaptive_retrieval_summary
        print(f"\nAdaptive Retrieval Summary:")
        print(f"  Arguments enhanced: {adaptive_retrieval_summary['arguments_enhanced']}")
        print(f"  Avg score improvement: {adaptive_retrieval_summary['average_score_improvement']:.3f}")

    state["validated_arguments"] = validated_arguments
    state["current_step"] = "final_answer_generation"

    avg_validity = (
        sum(getattr(arg, "validity_score", 0) for arg in validated_arguments) / len(validated_arguments)
        if validated_arguments else 0
    )
    print(f"\nValidation complete. Average validity: {avg_validity:.2f}")
    print(f"  High validity (>0.8): {sum(1 for arg in validated_arguments if getattr(arg, 'validity_score', 0) > 0.8)}")
    print(f"  Moderate (0.5-0.8): {sum(1 for arg in validated_arguments if 0.5 <= getattr(arg, 'validity_score', 0) <= 0.8)}")
    print(f"  Low validity (<0.5): {sum(1 for arg in validated_arguments if getattr(arg, 'validity_score', 0) < 0.5)}")

    return state

def final_answer_generator(state: GraphState) -> GraphState:
    print("[STEP] final_answer_generator: Synthesizing final answer for task:", state.get('task_name'))
    print("Generating final legal answer.")
    options = state.get("options", [])
    validated_arguments = state.get("validated_arguments", [])
    rag_context = state.get("rag_context", "")
    task_name = state.get("task_name", "")
    task_info = state.get("task_info", "")

    # Organize arguments by option and type
    arguments_by_option = {opt: {"support": [], "attack": []} for opt in options}
    for arg in validated_arguments:
        if arg.parent_option in arguments_by_option:
            arguments_by_option[arg.parent_option][arg.argument_type].append(arg)

    # Build prompt for LLM synthesis
    prompt = f"""
    You are a legal expert. Based on the validated arguments and relevant legal documents, synthesize a final answer for the following legal task:
    Task: {task_name}
    Case Information: {task_info}

    {rag_context}

    Options and Validated Arguments:
    """
    for option in options:
        prompt += f"\nOption: {option}"
        support_args = arguments_by_option[option]["support"]
        if support_args:
            prompt += "\n  Support arguments:"
            for arg in sorted(support_args, key=lambda x: getattr(x, "validity_score", 0) or 0.0, reverse=True):
                score = getattr(arg, 'validity_score', 0)
                if score is None:
                    score = 0.0
                prompt += f"\n    - [{score:.2f}] {arg.content}"
                if hasattr(arg, "supporting_docs") and arg.supporting_docs:
                    prompt += " (Evidence cited)"
        attack_args = arguments_by_option[option]["attack"]
        if attack_args:
            prompt += "\n  Challenge arguments:"
            for arg in sorted(attack_args, key=lambda x: getattr(x, "validity_score", 0) or 0.0, reverse=True):
                score = getattr(arg, 'validity_score', 0)
                if score is None:
                    score = 0.0
                prompt += f"\n    - [{score:.2f}] {arg.content}"

    prompt += """
    Based on the above, provide:
    1. The most legally justified answer (choose one option and explain why)
    2. A summary of supporting and challenging arguments for the chosen option
    3. Cite relevant legal documents using [REF-X] format where appropriate
    4. Briefly discuss any remaining uncertainties or limitations
    Return a clear, structured answer suitable for legal reasoning.
    """

    if state.get("enable_streaming", False):
        response_chunks = []
        for chunk in call_llm_stream(prompt, temperature=0.7, max_tokens=1024):
            response_chunks.append(chunk)
            state["final_answer_stream"] = "".join(response_chunks)
        response = "".join(response_chunks)
    else:
        response = call_llm(prompt, temperature=0.7, max_tokens=1024)

    # Extract cited document references
    cited_refs = re.findall(r"\[REF-(\d+)\]", response)
    cited_doc_ids = set(int(ref) for ref in cited_refs)
    if "document_references" in state:
        for ref in state["document_references"]:
            if ref["id"] in cited_doc_ids:
                ref["used_in"].append("final_answer")
                state["cited_documents"].add(ref["id"])

    state["final_answer"] = {
        "answer": response,
        "cited_document_ids": list(cited_doc_ids),
        "total_documents_retrieved": len(state.get("retrieved_documents", [])),
        "documents_cited": len(cited_doc_ids),
    }
    state["current_step"] = "complete"
    print("Final answer generated and stored in state.")
    return state