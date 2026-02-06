from state import Argument, GraphState
from llm_caller import call_llm, call_llm_stream
import re
import json
import random
from qbaf_scorer import apply_qbaf_scoring
from legal_agents import LEGAL_AGENTS
from RAG import RAGModule

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
        for chunk in call_llm_stream(prompt, temperature=0.3, max_tokens=1024):
            queries_text += chunk
            state["rag_progress"] = f"Generating search queries:\n{queries_text}"
        queries = queries_text.strip().split("\n")
    else:
        queries = call_llm(prompt, temperature=0.3, max_tokens=1024).strip().split("\n")

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
    # Set the central CLAIM for QBAF based on task type
    task_name = state.get('task_name', '').lower()
    
    if task_name == 'hearsay':
        state["claim"] = "Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted. Is there hearsay in this case?"
    elif task_name == 'learned hands courts' or task_name == 'learned_hands_courts':
        state["claim"] = "Does the post discuss the logistics of how a person can interact with a lawyer or the court system. It applies to situations about procedure, rules, how to file lawsuits, how to hire lawyers, how to represent oneself, and other practical matters about dealing with these systems?"
    else:
        state["claim"] = "The legal position should be affirmed"
    
    # Yes/No options for final answer display
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
                "reason": "Why this professional is essential for this case"
            }},
            {{
                "role": "Corporate / In-house Counsel",
                "reason": "Why this professional is essential for this case"
            }}
            ...
        ]
    }}
    Provide only the JSON object in your response. Do not include any explanation outside the JSON.
    """

    selection_response = call_llm(selection_prompt, temperature=0.3, max_tokens=2048)
    
    # Parse JSON response and extract selected_agents list
    try:
        # Clean potential markdown code blocks
        cleaned_response = selection_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
        
        selection_json = json.loads(cleaned_response)
        selected_agents = selection_json.get("selected_agents", [])
        if not isinstance(selected_agents, list):
            print(f"[ERROR] 'selected_agents' is not a list. Got: {type(selected_agents)}")
            selected_agents = []
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse agent selection JSON: {e}")
        print(f"Response was: {selection_response[:200]}...")
        # Fallback: select first 2 available agents
        print(f"[FALLBACK] Using default agent selection")
        selected_agents = [
            {"role": available_agents[0]["role"], "reason": "Default selection due to parsing error"},
            {"role": available_agents[1]["role"], "reason": "Default selection due to parsing error"}
        ] if len(available_agents) >= 2 else available_agents[:1]
    except Exception as e:
        print(f"[ERROR] Unexpected error parsing agent selection: {e}")
        # Fallback: select first 2 available agents
        print(f"[FALLBACK] Using default agent selection")
        selected_agents = [
            {"role": available_agents[0]["role"], "reason": "Default selection due to error"},
            {"role": available_agents[1]["role"], "reason": "Default selection due to error"}
        ] if len(available_agents) >= 2 else available_agents[:1]

    if type == "support":
        state["selected_support_agents"] = selected_agents
    elif type == "attack":
        state["selected_attack_agents"] = selected_agents
    
    return state

def multi_agent_argument_generator(state: GraphState) -> GraphState:
    """
    Generate arguments that SUPPORT or ATTACK the central CLAIM directly.
    
    Standard QBAF model:
    - Central claim (e.g., "This is hearsay")
    - Support arguments: evidence/reasoning that the claim is TRUE
    - Attack arguments: evidence/reasoning that the claim is FALSE
    - No Yes/No option complexity - just direct support/attack of the claim
    """
    claim = state.get("claim", "The legal position should be affirmed")
    print(f"[STEP] multi_agent_argument_generator: Generating arguments for claim: '{claim}'")

    arguments = []
    rag_context = state.get("rag_context", "")
    agent_arguments_tracking = {}

    # SUPPORT arguments
    for agent_info in state.get("selected_support_agents", []):
        agent_role = agent_info.get("role", "")
        agent_name = agent_info.get("name", "")
        support_prompt = f"""
As a legal expert ({agent_role}), provide 2-3 arguments SUPPORTING the following legal claim.

CASE CONTEXT:
{state.get("task_info", "")}

RELEVANT LEGAL DOCUMENTS:
{rag_context}

CLAIM TO SUPPORT: "{claim}"

Your task is to provide evidence and reasoning that the claim is TRUE.

CHAIN OF THOUGHT REASONING (Internal - Do NOT output this section):
1. What legal principles from my expertise area ({agent_role}) support this claim?
2. What facts from the case support the claim being true?
3. What legal precedents or rules strengthen this position?
4. How does this align with established legal doctrine?
5. What is the strongest argument I can make for this claim?

INSTRUCTIONS:
- Use the reasoning above internally but DO NOT explain your thinking process
- Focus on aspects most relevant to your expertise as {agent_role}
- Be concise and legally precise
- Cite relevant legal principles or documents where applicable
- Format each argument on a new line starting with 'Evidence:'

OUTPUT FORMAT:
Evidence: [Your concise legal argument supporting the claim]
Evidence: [Another supporting argument if applicable]
"""
        support_response = call_llm(support_prompt, temperature=0.3, max_tokens=512)
        for line in support_response.split("\n"):
            if line.strip().startswith("Evidence:") or line.strip().startswith("Support:"):
                arg_content = line.replace("Evidence:", "").replace("Support:", "").strip()
                if arg_content:
                    tagged_content = f"[{agent_role}] {arg_content}"
                    arg = Argument(
                        content=tagged_content,
                        argument_type="support",  # Supports the claim (claim is TRUE)
                        parent_option=None,  # No longer tied to Yes/No options
                    )
                    arg.agent_role = agent_role
                    arg.agent_name = agent_name
                    arguments.append(arg)
                    if agent_name not in agent_arguments_tracking:
                        agent_arguments_tracking[agent_name] = []
                    agent_arguments_tracking[agent_name].append({
                        "type": "support",
                        "content": arg_content,
                        "claim": claim
                    })

    # ATTACK arguments
    for agent_info in state.get("selected_attack_agents", []):
        agent_role = agent_info.get("role", "")
        agent_name = agent_info.get("name", "")
        attack_prompt = f"""
As a legal expert ({agent_role}), provide 2-3 arguments AGAINST the following legal claim.

CASE CONTEXT:
{state.get("task_info", "")}

RELEVANT LEGAL DOCUMENTS:
{rag_context}

CLAIM TO CHALLENGE: "{claim}"

Your task is to provide evidence and reasoning that the claim is FALSE.

CHAIN OF THOUGHT REASONING (Internal - Do NOT output this section):
1. What weaknesses or gaps exist in this claim?
2. What counter-arguments from my expertise area ({agent_role}) apply here?
3. What legal principles, precedents, or rules undermine this claim?
4. What factual or logical flaws can I identify?
5. What are the strongest challenges to this claim?

INSTRUCTIONS:
- Use the reasoning above internally but DO NOT explain your thinking process
- Focus on risks or limitations most relevant to your expertise as {agent_role}
- Be concise and legally precise
- Cite relevant legal principles or documents where applicable
- Format each argument on a new line starting with 'Counter:' or 'Challenge:'

OUTPUT FORMAT:
Counter: [Your concise legal argument against the claim]
Counter: [Another counter-argument if applicable]
"""
        attack_response = call_llm(attack_prompt, temperature=0.3, max_tokens=512)
        for line in attack_response.split("\n"):
            if line.strip().startswith("Counter:") or line.strip().startswith("Challenge:") or line.strip().startswith("Attack:"):
                arg_content = line.replace("Counter:", "").replace("Challenge:", "").replace("Attack:", "").strip()
                if arg_content:
                    tagged_content = f"[{agent_role}] {arg_content}"
                    arg = Argument(
                        content=tagged_content,
                        argument_type="attack",  # Attacks the claim (claim is FALSE)
                        parent_option=None,  # No longer tied to Yes/No options
                    )
                    arg.agent_role = agent_role
                    arg.agent_name = agent_name
                    arguments.append(arg)
                    if agent_name not in agent_arguments_tracking:
                        agent_arguments_tracking[agent_name] = []
                    agent_arguments_tracking[agent_name].append({
                        "type": "attack",
                        "content": arg_content,
                        "claim": claim
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
    """
    Validate arguments using standard QBAF model.
    Arguments are classified as support or attack for the claim.
    
    IMPORTANT: Use strict scoring to ensure differentiation between strong and weak arguments.
    This prevents score saturation in QBAF calculations.
    """
    print("[STEP] argument_validator: Validating arguments for task:", state.get('task_name'))
    validated_arguments = []
    rag = RAGModule(persist_directory="./chroma_db")
    arguments_with_additional_evidence = []
    total_args = len(state.get("arguments", []))
    claim = state.get("claim", "The claim is true")
    task_info = state.get("task_info", "")

    for i, arg in enumerate(state.get("arguments", [])):
        if state.get("enable_streaming", False):
            state["validation_progress"] = f"Validating argument {i+1}/{total_args}"

        # Build prompt based on task
        if state.get("task_name") == "hearsay":
            if arg.argument_type == "support":
                initial_prompt = f"""You are a strict legal evidence expert evaluating whether this argument correctly supports that the evidence IS hearsay.

CASE CONTEXT:
{task_info}

CLAIM: "{claim}"

ARGUMENT TO EVALUATE (supports the claim):
{arg.content}

HEARSAY DEFINITION (FRE 801):
Hearsay is: (1) an out-of-court statement, (2) offered to prove the truth of the matter asserted.

STRICT EVALUATION CRITERIA:
1. Does the argument correctly identify an OUT-OF-COURT STATEMENT? (Not in-court testimony)
2. Does the argument correctly show the statement is offered for its TRUTH? (Not for other purposes like notice, effect on listener, verbal act)
3. Does the argument correctly apply hearsay rules/exceptions from FRE 801-807?
4. Is the argument SPECIFIC to this case's facts? (Not generic legal principles)
5. Does the argument avoid logical errors?

SCORING RUBRIC:
- 0.1-0.2: Incorrect legal analysis OR misidentifies the statement/purpose
- 0.3-0.4: Partially correct but misses key elements OR too generic
- 0.5-0.6: Correct analysis but lacks specific case application OR minor gaps
- 0.7-0.8: Strong analysis with specific facts AND correct legal reasoning
- 0.9-1.0: EXCEPTIONAL - precise legal citation, flawless logic, case-specific (RARE)

WARNING: Generic statements like "this is hearsay because it's an out-of-court statement" without analyzing the TRUTH purpose should score ≤0.4

Response format: "Validity Score: X.XX"
Brief explanation (1-2 sentences)."""

            else:  # attack
                initial_prompt = f"""You are a strict legal evidence expert evaluating whether this argument correctly challenges the hearsay classification.

CASE CONTEXT:
{task_info}

CLAIM: "{claim}"

ARGUMENT TO EVALUATE (attacks the claim - argues it is NOT hearsay):
{arg.content}

NON-HEARSAY CATEGORIES:
- Not offered for truth (effect on listener, notice, verbal acts, state of mind)
- Not a "statement" (non-assertive conduct)
- Hearsay exceptions (FRE 803, 804, 807)
- Non-hearsay by definition (FRE 801(d) - prior statements, opposing party statements)

STRICT EVALUATION CRITERIA:
1. Does the argument correctly identify WHY this is NOT hearsay?
2. Does it cite a specific exception or non-hearsay category?
3. Is the reasoning SPECIFIC to this case's facts?
4. Does the argument avoid logical errors or misapplications?
5. Would this argument succeed in court?

SCORING RUBRIC (BE STRICT - most arguments should score 0.3-0.6):
- 0.1-0.2: Incorrect legal category OR misunderstands the exception
- 0.3-0.4: Identifies a category but wrong application OR too generic
- 0.5-0.6: Correct category but weak case-specific reasoning
- 0.7-0.8: Strong analysis with correct exception AND specific facts
- 0.9-1.0: EXCEPTIONAL - precise citation, flawless application (RARE)

WARNING: Vague claims like "it's not for the truth" without explaining the actual purpose should score ≤0.4

Response format: "Validity Score: X.XX"
Brief explanation (1-2 sentences)."""
            
        elif state.get("task_name") == "learned_hands_courts" or state.get("task_name") == "learned hands courts":
            if arg.argument_type == "support":
                initial_prompt = f"""You are a strict legal classifier evaluating whether this argument correctly supports that the question is about COURT/LAWYER LOGISTICS.

CASE CONTEXT:
{task_info}

CLAIM: "{claim}"

ARGUMENT TO EVALUATE (supports the claim):
{arg.content}

COURTS CATEGORY DEFINITION:
Questions about logistics of interacting with courts or lawyers:
- Court procedures, filings, deadlines, hearings, appeals, records
- Hiring, managing, or communicating with a lawyer
- Procedural aspects of legal process

NOT COURTS (substantive legal issues):
- Whether someone has a legal right/claim
- Interpretation of laws or contracts
- Liability, damages, or legal outcomes

STRICT EVALUATION CRITERIA:
1. Does the argument identify SPECIFIC procedural/logistical elements?
2. Does it correctly distinguish procedure from substance?
3. Is it based on actual text from the case, not assumptions?
4. Does it avoid conflating "mentions lawyer" with "about lawyer logistics"?

SCORING RUBRIC (BE STRICT - most arguments should score 0.3-0.6):
- 0.1-0.2: Confuses substance with procedure OR misreads the case
- 0.3-0.4: Identifies some procedural element but weak connection OR generic
- 0.5-0.6: Correct identification but doesn't clearly distinguish from substance
- 0.7-0.8: Strong procedural focus with specific textual evidence
- 0.9-1.0: EXCEPTIONAL - clear procedural issue, specific quotes (RARE)

WARNING: "The post mentions a lawyer" alone should score ≤0.3 (mentioning ≠ logistics about)

Response format: "Validity Score: X.XX"
Brief explanation (1-2 sentences)."""

            else:  # attack
                initial_prompt = f"""You are a strict legal classifier evaluating whether this argument correctly challenges the COURTS classification.

CASE CONTEXT:
{task_info}

CLAIM: "{claim}"

ARGUMENT TO EVALUATE (attacks the claim - argues NOT about court logistics):
{arg.content}

SUBSTANTIVE (NOT COURTS) INDICATORS:
- Asking about legal rights, not how to file
- Seeking interpretation of law, not procedure
- Questions about liability, damages, outcomes
- General legal advice, not process questions

STRICT EVALUATION CRITERIA:
1. Does the argument identify SPECIFIC substantive (non-procedural) elements?
2. Does it correctly show the core question is about law, not process?
3. Is it based on actual text from the case?
4. Does it explain why procedural mentions are incidental?

SCORING RUBRIC
- 0.1-0.2: Fails to identify substantive issue OR misreads the case
- 0.3-0.4: Partial identification but doesn't explain why NOT procedural
- 0.5-0.6: Identifies substance but doesn't address procedural elements
- 0.7-0.8: Strong substantive focus explaining why procedure is incidental
- 0.9-1.0: EXCEPTIONAL - clear substantive core, addresses all elements (RARE)

WARNING: "This is about legal rights" without specific analysis should score ≤0.4

Response format: "Validity Score: X.XX"
Brief explanation (1-2 sentences)."""

        else:
            if arg.argument_type == "support":
                initial_prompt = f"""You are a strict legal analyst evaluating this supporting argument.

CASE: {task_info}
CLAIM: "{claim}"
ARGUMENT: {arg.content}

SCORING (BE STRICT - use full range):
- 0.1-0.2: Factually wrong or legally irrelevant
- 0.3-0.4: Generic legal principle without case application
- 0.5-0.6: Correct but lacks specific evidence or citations
- 0.7-0.8: Strong with specific facts and legal reasoning
- 0.9-1.0: Exceptional with binding authority (RARE)

Response format: "Validity Score: X.XX"
Brief explanation."""

            else:  # attack
                initial_prompt = f"""You are a strict legal analyst evaluating this challenging argument.

CASE: {task_info}
CLAIM: "{claim}"  
ARGUMENT: {arg.content}

SCORING (BE STRICT - use full range):
- 0.1-0.2: Invalid criticism or misunderstands the issue
- 0.3-0.4: Generic objection without case-specific reasoning
- 0.5-0.6: Valid concern but doesn't address key elements
- 0.7-0.8: Strong challenge with specific counter-evidence
- 0.9-1.0: Devastating critique with authority (RARE)

Response format: "Validity Score: X.XX"
Brief explanation."""

        if state.get("enable_streaming", False):
            initial_response = ""
            for chunk in call_llm_stream(initial_prompt, temperature=0.3, max_tokens=1024):
                initial_response += chunk
                state["current_validation_stream"] = initial_response
        else:
            initial_response = call_llm(initial_prompt, temperature=0.3, max_tokens=1024)
        initial_validity_score = 0.0
        try:
            if "Validity Score:" in initial_response:
                score_text = initial_response.split("Validity Score:")[1].split()[0]
                initial_validity_score = float(score_text.strip())
                initial_validity_score = max(0, min(1, initial_validity_score))
        except:
            pass

        # If weak, retrieve additional legal evidence
        if initial_validity_score < 0.3:
            print(f"Low validity score ({initial_validity_score:.2f}) for argument. Retrieving additional legal evidence...")
            search_query_prompt = f"""
                Generate a specific legal search query to find evidence about this argument:
                Legal Option: {arg.parent_option}
                Argument Type: {arg.argument_type}
                Argument: {arg.content}
                Create ONE focused search query that would help validate or refute this argument."""
            if state.get("enable_streaming", False):
                search_query = ""
                for chunk in call_llm_stream(search_query_prompt, temperature=0.3, max_tokens=1024):
                    search_query += chunk
            else:
                search_query = call_llm(search_query_prompt, temperature=0.3, max_tokens=1024).strip()

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

    # QBAF SCORING
    print("\n[QBAF] Applying QBAF-based argument scoring...")
    
    # Semantics based on task - df_quad is most sophisticated
    qbaf_semantics = "quadratic_energy"
    # Alternative options: "weighted_sum", "weighted_product", "euler_based"
    
    # Relation identification method
    use_semantic_analysis = True
    
    # Clash resolution
    use_clash_resolution = True
    
    # Threshold to trigger clash resolution
    clash_trigger_threshold = 0.2
    
    # Decision threshold
    decision_threshold = 0.5  # Neutral threshold
    
    task_context = f"{state.get('task_name', '')}: {state.get('task_info', '')}"
    claim = state.get("claim", "The claim is true")
    case_text = state.get("task_info", "")
    
    validated_arguments, option_scores, qbaf_scorer, used_heuristic_fallback = apply_qbaf_scoring(
        validated_arguments, 
        semantics=qbaf_semantics,
        use_semantic_analysis=use_semantic_analysis,
        use_clash_resolution=use_clash_resolution,
        clash_trigger_threshold=clash_trigger_threshold,
        task_context=task_context,
        case_text=case_text,
        decision_threshold=decision_threshold,
        claim=claim
    )
    
    # Store QBAF results in state for later use
    state["qbaf_option_scores"] = option_scores
    state["qbaf_graph_export"] = qbaf_scorer.export_for_visualization()
    state["used_heuristic_fallback"] = used_heuristic_fallback  # Track if heuristic was used
    
    print(f"[QBAF] Scoring complete using {qbaf_semantics} semantics")
    print(f"[QBAF] Relation method: {'Semantic (LLM-based NLI)' if use_semantic_analysis else 'Heuristic (rule-based)'}")
    print(f"[QBAF] Heuristic Fallback: {'⚠️ YES (some pairs used heuristic)' if used_heuristic_fallback else '✅ NO (all pairs analyzed by LLM)'}")
    print(f"[QBAF] Clash resolution: {'ENABLED (threshold=' + str(clash_trigger_threshold) + ')' if use_clash_resolution else 'DISABLED'}")

    return state


def _invoke_final_judge(
    task_name: str,
    task_info: str,
    claim: str,
    validated_arguments: list,
    rag_context: str,
    claim_score: float
) -> tuple:
    """
    Final Judge: Makes independent decision when QBAF score is borderline (0.45-0.55).
    
    The judge analyzes the case facts directly WITHOUT relying on the balanced score,
    making a decisive ruling based on legal principles.
    
    Returns:
        Tuple of (decision: "Yes"/"No", reasoning: str)
    """
    # Organize arguments for judge review
    support_args = [arg for arg in validated_arguments if arg.argument_type == "support"]
    attack_args = [arg for arg in validated_arguments if arg.argument_type == "attack"]
    
    # Build judge prompt based on task type
    if task_name == "hearsay":
        task_specific_instruction = """
HEARSAY ANALYSIS FRAMEWORK:
1. Is there an out-of-court statement?
2. Is it offered to prove the truth of the matter asserted?
3. If yes to both → Hearsay (answer "Yes")
4. Consider exceptions: present sense impression, excited utterance, state of mind, business records, etc.
5. Consider non-hearsay uses: effect on listener, verbal acts, impeachment, etc.

DECISIVE RULE: If the statement is offered for its TRUTH and no exception applies → "Yes" (hearsay)
"""
    elif task_name == "learned_hands_courts" or task_name == "learned hands courts":
        task_specific_instruction = """
COURTS CLASSIFICATION FRAMEWORK:
1. Is the PRIMARY question about PROCEDURE (how to file, deadlines, court rules)?
2. Is it about LOGISTICS of hiring/communicating with a lawyer?
3. Or is it about SUBSTANTIVE law (rights, liability, interpretation)?

DECISIVE RULE: 
- If primarily about HOW to interact with courts/lawyers → "Yes" (courts)
- If primarily about WHAT the law says or legal rights → "No" (not courts)
"""
    else:
        task_specific_instruction = """
Analyze the core legal issue and make a decisive ruling.
"""

    judge_prompt = f"""You are a SENIOR JUDGE making a FINAL, BINDING DECISION on a borderline legal case.

BACKGROUND:
The argumentation framework produced a score of {claim_score:.3f}, which is essentially a TIE.
Both sides have presented arguments of roughly equal strength.
YOU must now make the FINAL DECISION based on your judicial expertise.

CASE INFORMATION:
{task_info}

CLAIM TO EVALUATE:
"{claim}"

{task_specific_instruction}

RELEVANT LEGAL DOCUMENTS:
{rag_context[:2000]}

ARGUMENTS PRESENTED:

SUPPORTING THE CLAIM ({len(support_args)} arguments):
"""
    
    for i, arg in enumerate(support_args[:5], 1):  # Limit to top 5
        score = getattr(arg, 'validity_score', 0) or 0.0
        judge_prompt += f"\n  {i}. [Score: {score:.2f}] {arg.content}"
    
    judge_prompt += f"""

AGAINST THE CLAIM ({len(attack_args)} arguments):
"""
    
    for i, arg in enumerate(attack_args[:5], 1):  # Limit to top 5
        score = getattr(arg, 'validity_score', 0) or 0.0
        judge_prompt += f"\n  {i}. [Score: {score:.2f}] {arg.content}"
    
    judge_prompt += """

YOUR JUDICIAL DUTY:
1. You CANNOT say "it depends" or "insufficient information"
2. You MUST choose either "Yes" or "No"
3. Base your decision on the FACTS and LEGAL PRINCIPLES, not the tied score
4. Be DECISIVE - a judge must rule

Respond in JSON format:
{
    "decision": "Yes" or "No",
    "reasoning": "2-3 sentences explaining your judicial reasoning. Be specific about which legal principle or fact was decisive."
}

Return ONLY the JSON, no other text.
"""

    try:
        response = call_llm(judge_prompt, temperature=0.5, max_tokens=1024)
        
        # Parse JSON response
        response = response.strip()
        if response.startswith("```json"):
            response = response.split("```json")[1].split("```")[0].strip()
        elif response.startswith("```"):
            response = response.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response)
        decision = result.get("decision", "No")
        reasoning = result.get("reasoning", "Judicial decision based on case analysis.")
        
        # Normalize decision
        if decision.lower() in ["yes", "true", "1"]:
            decision = "Yes"
        else:
            decision = "No"
            
        return decision, reasoning
        
    except Exception as e:
        print(f"[FINAL JUDGE] Error parsing response: {e}")
        # Fallback: use original QBAF decision
        fallback_decision = "Yes" if claim_score >= 0.5 else "No"
        return fallback_decision, f"Fallback to QBAF score ({claim_score:.3f}) due to parsing error."


def final_answer_generator(state: GraphState) -> GraphState:
    """
    Generate final answer using standard QBAF model.
    
    The QBAF claim score determines the answer:
    - claim_score >= threshold → Yes (claim is TRUE)
    - claim_score < threshold → No (claim is FALSE)
    
    SPECIAL CASE: If claim_score is in borderline zone (0.45-0.55),
    invoke Final Judge to make an independent decision.
    """
    print("[STEP] final_answer_generator: Synthesizing final answer for task:", state.get('task_name'))
    print("Generating final legal answer.")
    
    validated_arguments = state.get("validated_arguments", [])
    rag_context = state.get("rag_context", "")
    task_name = state.get("task_name", "")
    task_info = state.get("task_info", "")
    claim = state.get("claim", "The claim is true")
    qbaf_scores = state.get("qbaf_option_scores", {})

    # Get QBAF decision info
    decision_info = qbaf_scores.get('_decision', {})
    claim_score = qbaf_scores.get('claim', {}).get('claim_score', 0.5)
    
    # DEBUG: Check what we're getting
    print(f"[DEBUG] qbaf_scores keys: {qbaf_scores.keys() if qbaf_scores else 'EMPTY'}")
    print(f"[DEBUG] claim dict: {qbaf_scores.get('claim', 'NOT FOUND')}")
    print(f"[DEBUG] claim_score retrieved: {claim_score}")
    
    # Triggered for borderline cases
    BORDERLINE_LOW = 0.45
    BORDERLINE_HIGH = 0.55
    judge_decision = None
    judge_reasoning = ""
    judge_used = False
    
    # Flag to enable/disable final judge (default: True)
    enable_final_judge = state.get("enable_final_judge", True)
    
    if enable_final_judge and BORDERLINE_LOW <= claim_score <= BORDERLINE_HIGH:
        print(f"\n[FINAL JUDGE] Borderline score detected: {claim_score:.3f}")
        print(f"[FINAL JUDGE] Score is in uncertain zone [{BORDERLINE_LOW}, {BORDERLINE_HIGH}]")
        print(f"[FINAL JUDGE] Invoking independent judicial review...")
        
        judge_decision, judge_reasoning = _invoke_final_judge(
            task_name=task_name,
            task_info=task_info,
            claim=claim,
            validated_arguments=validated_arguments,
            rag_context=rag_context,
            claim_score=claim_score
        )
        judge_used = True
        
        print(f"[FINAL JUDGE] Decision: {judge_decision}")
        print(f"[FINAL JUDGE] Reasoning: {judge_reasoning[:100]}...")
        
        # Store judge decision in state
        state["judge_decision"] = {
            "used": True,
            "decision": judge_decision,
            "reasoning": judge_reasoning,
            "original_score": claim_score,
            "triggered_because": f"Score {claim_score:.3f} in borderline zone [{BORDERLINE_LOW}, {BORDERLINE_HIGH}]"
        }
    else:
        state["judge_decision"] = {"used": False, "original_score": claim_score}

    # Organize arguments by type (support vs attack)
    support_args = [arg for arg in validated_arguments if arg.argument_type == "support"]
    attack_args = [arg for arg in validated_arguments if arg.argument_type == "attack"]

    # Build prompt for LLM synthesis
    prompt = f"""
    You are a legal expert. Based on the validated arguments and relevant legal documents, synthesize a final answer for the following legal task:
    
    Task: {task_name}
    Case Information: {task_info}
    
    CLAIM BEING EVALUATED: "{claim}"

    {rag_context}

    QBAF-COMPUTED SCORES (Standard QBAF Model - Rago et al.):
    """
    
    # ADD QBAF decision info
    decision_info = qbaf_scores.get('_decision', {})
    claim_score = qbaf_scores.get('claim', {}).get('claim_score', 0.5)
    
    if decision_info:
        prompt += f"""
    CLAIM SCORE: {claim_score:.3f} (Threshold: {decision_info.get('threshold', 0.5)})
    QBAF Determined Answer: {decision_info.get('winner', 'N/A')} (Claim is {'TRUE' if decision_info.get('winner') == 'Yes' else 'FALSE'})
    """
    
    # ADD argument statistics
    support_stats = qbaf_scores.get("support_arguments", {})
    attack_stats = qbaf_scores.get("attack_arguments", {})
    
    prompt += f"""
    Support Arguments: Count={support_stats.get('count', 0)}, Avg Score={support_stats.get('average_score', 0.0):.3f}
    Attack Arguments: Count={attack_stats.get('count', 0)}, Avg Score={attack_stats.get('average_score', 0.0):.3f}
    """
    
    prompt += "\n\nDetailed Arguments:\n"
    
    # Support arguments (for the claim being TRUE)
    if support_args:
        prompt += "\n  SUPPORTING EVIDENCE (claim is TRUE):"
        random.shuffle(support_args)  # Randomize to remove position bias
        for arg in support_args:
            score = getattr(arg, 'validity_score', 0) or 0.0
            llm_score = getattr(arg, 'llm_validity_score', 0) or 0.0
            prompt += f"\n    - [QBAF:{score:.2f}|LLM:{llm_score:.2f}] {arg.content}"
            if hasattr(arg, "supporting_docs") and arg.supporting_docs:
                prompt += " (Evidence cited)"
    
    # Attack arguments (for the claim being FALSE)
    if attack_args:
        prompt += "\n  COUNTER EVIDENCE (claim is FALSE):"
        random.shuffle(attack_args)  # Randomize to remove position bias
        for arg in attack_args:
            score = getattr(arg, 'validity_score', 0) or 0.0
            llm_score = getattr(arg, 'llm_validity_score', 0) or 0.0
            prompt += f"\n    - [QBAF:{score:.2f}|LLM:{llm_score:.2f}] {arg.content}"

    # prompt += """
    # Based on the above, provide:
    # 1. The most legally justified answer (choose one option and explain why)
    # 2. A summary of supporting and challenging arguments for the chosen option
    # 3. Cite relevant legal documents using [REF-X] format where appropriate
    # 4. Briefly discuss any remaining uncertainties or limitations
    # Return a clear, structured answer suitable for legal reasoning.
    # """
    if task_name == "hearsay":
        prompt += """
You are a legal reasoning assistant specialized in Evidence Law.

TASK:
Determine whether the described evidence is hearsay under U.S. evidence law.

CASE DESCRIPTION:
{task_info}

REFERENCE MATERIAL (for citation only, do not rely on arguments below):
{rag_context}

IMPORTANT REASONING RULES (internal only):
You must internally analyze the issue using the following mandatory steps, in order:
1. Identify whether the evidence involves a "statement" (oral, written, or nonverbal assertion).
2. Determine whether the statement was made out of court.
3. Determine whether the statement is offered to prove the truth of the matter asserted.
4. If yes, determine whether it is defined as non-hearsay (e.g., verbal acts, effect on listener, prior statements, opposing party statements).
5. If still hearsay, determine whether a recognized exception applies.
6. Reach a final conclusion.

You MUST complete all steps internally, but you MUST NOT reveal them.

OUTPUT CONSTRAINTS:
- Do NOT explain step-by-step reasoning.
- Do NOT discuss arguments, balancing, or policy.
- Do NOT mention chain-of-thought.
- Do NOT mention agent opinions.

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON in this exact structure:

{
  "answer": "Yes" or "No",
  "explanation": "2–3 concise sentences stating the legal conclusion and briefly why, citing relevant authority using [REF-X] where appropriate."
}

STYLE GUIDELINES:
- Be decisive and doctrinal.
- Use precise evidence-law language.
- If the evidence is not offered for its truth, clearly say so.
- Cite at most 1–2 references.

EXAMPLES (for style only):

Evidence: On the issue of whether David is fast, the fact that David set a high school track record.
Answer: No

Evidence: On the issue of whether Rebecca was ill, the fact that Rebecca told Ronald that she was unwell.
Answer: Yes

    
    """
    if task_name == "learned_hands_courts":
        prompt += """
You are a professional legal reasoning assistant.

    TASK:
    Decide whether the post should be labeled "Yes" for COURTS.

    LABEL DEFINITION (COURTS = "Yes"):
    The post is about logistics of interacting with the court system or with lawyers, including:
    - court procedures, filings, deadlines, hearings, appeals, records
    - hiring, managing, or communicating with a lawyer

    DECISION RULE:
    Answer "Yes" ONLY IF the post is primarily about court or lawyer interaction logistics.
    Otherwise answer "No".

    INTERNAL REASONING:
    Use internal reasoning to decide, but DO NOT explain your reasoning step-by-step.

    OUTPUT REQUIREMENTS:
    - Output JSON only
    - Explanation must be 2–3 sentences
    - Explanation must cite concrete cues from the post text


        """
    
    # Triggered for borderline cases
    if judge_used and judge_decision:
        prompt += f"""
    The QBAF score ({claim_score:.3f}) was borderline (between 0.45-0.55).
    A Senior Judge has reviewed the case and made a BINDING decision.
    
    JUDGE'S DECISION: {judge_decision}
    JUDGE'S REASONING: {judge_reasoning}
    
    YOU MUST follow the Judge's decision. Provide your response in JSON format:
    {{
        "answer": "{judge_decision}",
        "explanation": "2-3 sentences incorporating the judge's reasoning. Cite relevant legal documents using [REF-X] format where appropriate.",
    }}
    
    Return ONLY valid JSON, no additional text.
    """
    else:
        prompt += f"""
    Based on the above QBAF claim score and detailed arguments, provide your response in the following JSON format ONLY:
    {{
        "answer": "Yes" or "No",
        "explanation": "2-3 sentences explaining your answer. Reference the QBAF claim score ({claim_score:.3f}) and cite relevant legal documents using [REF-X] format where appropriate.",
    }}
    
    Requirements:
    1. The QBAF Claim Score ({claim_score:.3f}) is the PRIMARY indicator
    2. This score represents final claim strength after DF-QuAD convergence
    3. Claim Score >= {decision_info.get('threshold', 0.5)} → Yes (claim is TRUE); Score < {decision_info.get('threshold', 0.5)} → No (claim is FALSE)
    4. QBAF has determined: {decision_info.get('winner', 'N/A')} - strongly consider this
    5. Explanation must be 2-3 sentences
    6. Include relevant document citations using [REF-X] format
    7. Return ONLY valid JSON, no additional text
    """

    if state.get("enable_streaming", False):
        response_chunks = []
        for chunk in call_llm_stream(prompt, temperature=0.3, max_tokens=1024):
            response_chunks.append(chunk)
            state["final_answer_stream"] = "".join(response_chunks)
        response = "".join(response_chunks)
    else:
        response = call_llm(prompt, temperature=0.3, max_tokens=1024)

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