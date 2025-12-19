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
        For every question, you must internally run a structured chain-of-thought analysis, but **never reveal it**.
Only output the final JSON answer.
### Internal reasoning steps (not shown to user):
1. Determine whether there is a statement.
2. Determine whether the statement is out-of-court.
3. Determine whether it is offered to prove the truth of the matter asserted.
4. If it is hearsay, check whether it is excluded.
5. If still hearsay, evaluate applicable exceptions or exceptions requiring unavailability.
6. Conclude whether the evidence is hearsay or not.

### Your response behavior:

* Never reveal chain-of-thought reasoning.
* Only return the final conclusion + short explanation in JSON.
* Follow the style demonstrated in the examples.

### Format for all answers (Few-shot samples):
Q: On the issue of whether David is fast, the fact that David set a high school track record. Is there hearsay?
A: No

Q: On the issue of whether Rebecca was ill, the fact that Rebecca told Ronald that she was unwell. Is there hearsay?
A: Yes

Q: To prove that Tim was a soccer fan, the fact that Tim told Jimmy that "Real Madrid was the best soccer team in the world." Is there hearsay?
A: No

Q: When asked by the attorney on cross-examination, Alice testified that she had "never seen the plaintiff before, and had no idea who she was." Is there hearsay?
A: No

Q: On the issue of whether Martin punched James, the fact that Martin smiled and nodded when asked if he did so by an officer on the scene. Is there hearsay?
A: Yes

    
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
 blob:https://www.facebook.com/bb34055f-5ddd-4a52-bf0e-b0cade031942   - Explanation must cite concrete cues from the post text

    FEW-SHOT EXAMPLES:

    Post: Is it normal for a California court to fudge up your records?   I was supposed to have 2 previous arrests "dismissed" via expungement and well, they never did I guess. It took me being denied a really good job 3 years later to figure this out too.   I have the court documents that say my charges have here by been dismissed but my DOJ record is saying something different. What do I do? Do I have any sort of recourse against the court house for not getting that job. (I'm sure I don't, I'm just pretty broken up about it)  Thanks Legal Advice!
    Label: Yes

    Post: As a preface to this story I drive a truck for the nation's largest retailer and they pay me very good (six figures good).   I was on my off day and took an out of town trip with my wife and kids in the family minivan. We started to approach the small town of Uniontown AL and I was immediately pulled over by a police officer (he passed me going the opposite direction then proceeded to do a unturn to get behind me). I'm puzzled because I always drive safe and legal, my wife always gripes at me for this habit. Before the officer approached my window I had done decided that I must have hit a 45 mph zone without realizing it and I never slowed down from 55 mph. When he approached my window the first thing the officer told me is I was doing 70 in a 55. My jaw dropped because I know I was going 55. I didn't argue with the officer and I didn't confirm or deny doing 70. I pretty much remained silent. He gave me my citation and I carried on. My wife did research on this town's police force and found a message board with so many other horror stories just like mine with people getting pulled over for doing 70 in a 55. So I think it's obvious that this town is running a scam.   I know the obvious answer is to "lawyer up" and I'm planning on doing that this Monday. But my most concerning question is, will that even do any good in a corrupt jurisdiction? I mean if the cops are going to flat out lie and say motorists were speeding then does that mean the staff of the municipal court could be in on it as well?   And to revisit my beginning statement where I drive truck for the nation's largest retailer... They will not tolerate a serious violation on my mvr. So I'm possibly facing a $100,000/yr job loss. Who knows, they might be understanding and give me a slap on the wrist but either way I'm scared!   Where do I go from here?
    Label: Yes

    Post: Yesterday I received 3 notices of traffic tickets from Florence (Originally perpetrated in March) where I "was driving within the limited traffic area without authorization". Two infractions were within 40 minutes, and the other was 2 days later. I don't recall seeing signs, or being on a road that seemed like I shouldn't, but I don't speak Italian either!   I am an American citizen living in the US that was using a rental car. It's important to note that these notices seem to be from the city police directly, and not through the rental company. All 3 notices are for 81 + 37 Euros for procedures.   It ends up being $403 USD. What are my options and what would you recommend? Can I just not pay? Will I not be allowed to go back to Italy?
    Label: Yes

    Post: Im being asked to sign a custody agreement. Im being told that I can bring this matter back to the court in a year. However, in big bold letters and a lager font than the rest of the document it reads "THIS AGREEMENT IS NOT SUBJECT TO REVOCATION" under that it says As part of the consideration herein, tge parties acknowledge that this agreement is binding and not subject to revocation.   My ex whom im having a hard time trusting said that was put in for me, so visitation can never be withheld.  can someone please help me understand if i sign this agreement, would I be able to change it in 12 mos?
    Label: No

    Post: I made friends with a fellow nursing student about a year ago. We got along well for a most of that time until she started acting very strangely so I tried to reduce our interactions to more of an acquaintance level. I still helped her with school work and coached her through some NCLEX practice exams. I made sure to be very non-confrontational as she has mentioned all sorts of mental instability issues in the past and how she hates having any sort of direct confrontation. I have no idea if any claims about depression or anxiety are actually diagnosed.  We ended up with pretty much the SAME EXACT schedule this past semester. So naturally, we would walk to class together and occasionally go to the restroom together. Normal female things. Nothing she did gave me any cause for alarm. I never tried to bother her. I thought it was a simple, functional, school-based friendship.  This is where things get strange. She started acting very distant and would ignore about half the time when I would say hi and she started leaving class for long periods. Then she started leaving whenever I would even be in the same hall way. She'd leave class about a minute before it was over and not make eye contact with anyone. I thought maybe she was having a serious problem so I texted her to see if she was okay. If anything was wrong, etc. If I could help. I may not want to be best friends but when someone acts THAT oddly, maybe just a little compassion can change whatever situation they are in.  She assures me that everything is fine and then started ignoring me completely a few days later. I gave up on trying to figure out what was up or if I could help. I kept to myself. This was a while back.  I get a text about a week ago saying that she feels uncomfortable with how much I stalked her last semester and she didn't think it was okay for me to go to the restroom with her because she felt that I wanted an intimate relationship and would try something.  I told her I was uncomfortable with further contact after telling her it had been a while since I 'stalked' her and that the stalking she was talking about was unavoidable on both of our parts because we had five classes together in the same day. If she was following me or I her, it couldn't have been avoided but that whatever she perceived to be the case was simply us walking to class together. She was the one that showed up on campus on days that she had no classes and went to some of my classes with me (I had seven classes total that semester). I feel like if anyone should be scared of the situation, it should be me. I'm not, though. Just uncomfortable.  I'm worried that since this contact was so out of the blue and so unprovoked with specific details on how I made her feel threatened basically, that she's trying to set me up in some way. I'm genuinely worried because what if she's saying all of that and being so unpredictable because she's about to do something like file a restraining order? That would kill any future I could have as a nurse.  We are both early thirties in Nebraska if that affects regional laws.  What should I do? Save the texts and get a lawyer on retainer just in case? I don't understand why she would just now start doing all of this when we haven't really talked since last semester. This whole situation feels very wrong to me. I know she's a very good manipulator which is one of the main reasons I stopped associating with her so much. This all feels too odd and specific to mean anything good.  **TL;DR-** Ex-class mate from last semester texts me out of the blue saying that she feels like I was stalking her. Why bring this up now, without ever indicating that feeling before? What should I do to protect myself and my future?
    Label: No

    Post: I am trying to divorce my abusive spouse. We've been married 12 years and have a six year old daughter. He has been physically violent for years and has continued to stalk me pretty aggressively throughout our separation. I moved out in Nov. 2016 and officially filed with the courts in Feb. 2017 pro se. He counter petitioned and as I did not have an attorney I was bullied into giving majority custody to him of our daughter. This has proved to be a bad decision as he does not care for her emotional well being and she is now seeing a therapist. He also continues to break the morality clause in our custody orders. I have gone to authorities to attempt to get a Protective Order against him, once with a bruise from him on my body, and they stated that as we are still married and have a child together they thought it best to deny. I have had to fight to get a criminal trespass warning for my home. Since I decided to leave I have been seeing someone else. My boyfriend has been hit multiple times and charges were pressed but they never went anywhere. The local police told him he should just stop seeing me and that I'm getting what I deserve bc I shouldn't be seeing someone while I'm still technically married. On top of physical stalking, he works in IT security and cyber stalks me constantly. He hacks into ALL of my online accounts, wipes or locks out my phone (which he is not on the account or any of my accounts), gets into my bank accounts, paypal etc, and has begun hacking into the network where I work and messing with my payroll stuff as well as posting inappropriate pictures for all of my coworkers to see. Meanwhile, no one will listen to me or acknowledge whats going on no matter what I report. He even reads all my text messages, etc, tracks my whereabouts constantly, and does petty things like driving by my house and setting of my car alarm in the middle of the night. He has recently been diagnosed as having borderline personality disorder after a suicide attempt and I feel he is unfit to parent our child. However, I cant afford a retainer for an attorney. I now pay approximately 45% of my income to him for child support, insurance for our daughter, daycare for her while shes with him, and other medical fees each month. My check is already small so this has left me on the brink of homelessness even though I work full time and clean houses on the side. He tells our child awful things like I don't love her and I'm a quote "fucking whore" etc constantly which has been very detrimental to her. At this point I'm at my wits end. I feel like no one is listening and I have no where to turn. I would love to free myself and my daughter from this abuse and unsustainable situation, but I don't know what to do. Could anyone tell me how long he can ride this out without signing divorce papers since he counter petitioned? What can I do to lower the amount I pay to him as I am in such dire straits and he makes a few thousand a month more than I do already? Without everything coming out I currently bring home less than a minimum wage employee and cant find an attorney willing to help
    Label: No
        """
    
    
    prompt += f"""
    Based on the above, provide your response in the following JSON format ONLY:
    {{
        "answer": "Yes" or "No",
        "explanation": "2-3 sentences explaining your answer and citing relevant legal documents using [REF-X] format where appropriate."
    }}
    
    Requirements:
    1. Choose the most legally justified answer (Yes or No based on the options: {', '.join(options)})
    2. Explanation must be 2-3 sentences
    3. Include relevant document citations using [REF-X] format
    4. Return ONLY valid JSON, no additional text
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