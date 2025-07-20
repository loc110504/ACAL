"""
Hearsay Snippet Analyzer with Gradio Interface

This tool lets you upload or paste a single text snippet to check
whether it contains hearsay, using AI-driven semantic analysis
with human-in-the-loop validation via Gradio.

When launched, the web interface allows you to:
1. Paste or upload one passage of legal text
2. View AI‑flagged hearsay segments and their justification
3. Confirm, edit, or override each flag
4. Get the annotated snippet with explanatory notes

Note: What is hearsay?
“Hearsay” means a statement that:
(1) the declarant does not make while testifying at the current trial or hearing; and
(2) a party offers in evidence to prove the truth of the matter asserted in the statement.

Statements That Are Not Hearsay. A statement that meets the following conditions is not hearsay:
(1) A Declarant-Witness’s Prior Statement. The declarant testifies and is subject to cross-examination about a prior statement, and the statement:
    (A) is inconsistent with the declarant’s testimony and was given under penalty of perjury at a trial, hearing, or other proceeding or in a deposition;
    (B) is consistent with the declarant’s testimony and is offered:
        (i) to rebut an express or implied charge that the declarant recently fabricated it or acted from a recent improper influence or motive in so testifying ; or
        (ii) to rehabilitate the declarant's credibility as a witness when attacked on another ground ; or
    (C) identifies a person as someone the declarant perceived earlier.
(2) An Opposing Party’s Statement . The statement is offered against an opposing party and:
    (A) was made by the party in an individual or representative capacity;
    (B) is one the party manifested that it adopted or believed to be true;
    (C) was made by a person whom the party authorized to make a statement on the subject;
    (D) was made by the party’s agent or employee on a matter within the scope of that relationship and while it existed; or
    (E) was made by the party’s coconspirator during and in furtherance of the conspiracy.

"""
from typing import TypedDict, List, Dict, Optional
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
import torch
import gradio as gr
from env_config import HUGGINGFACE_MODEL_NAME
from llm_caller import call_huggingface_llm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
call_llm = call_huggingface_llm  # Use HuggingFace model for LLM calls


# State Definition
@dataclass
class Argument:
    """Represents an argument with its content, type, and validity score"""

    content: str
    argument_type: str  # "support" or "attack"
    validity_score: Optional[float] = None
    parent_option: Optional[str] = None


class GraphState(TypedDict):
    """State schema for the LangGraph workflow"""

    input_text: str
    handling_options: List[str]
    arguments: List[Argument]
    validated_arguments: List[Argument]
    revised_answer: Dict[str, any]
    human_feedback: Optional[str]
    current_step: str
    human_review_complete: bool  # Added for Gradio integration
    user_action: Optional[str]  # Added for storing user's choice


# Node Functions
def care_plan_generator(state: GraphState) -> GraphState:
    """First LLM: Identify potential hearsay segments in the text snippet"""

    prompt = f"""You are a legal AI assistant specialized in hearsay detection.
    Given the following text snippet, identify 1–3 segments (sentences or clauses)
    that may constitute hearsay under Rule 801(c) of the Federal Rules of Evidence.

    Text Snippet:
    {state['input_text']}

    Please provide handling options in the following format:
    Segment 1: [Description]
    Segment 2: [Description]
    ...
    
    """

    response = call_llm(prompt, temperature=0.5, max_tokens=1024)

    # Parse handling options
    segments = []
    for line in response.split("\n"):
        if line.strip().startswith("Segment"):
            segment_text = line.split(":", 1)[1].strip() if ":" in line else line
            if segment_text:
                segments.append(segment_text)

    # Ensure we have at least one segment
    if not segments:
        print("Warning: No segments parsed from LLM response. Using default segments.")
        segments = [
            "Out‑of‑court statements",
            "Assertions offered to prove the truth of the matter asserted",
            "Declarations by persons not testifying under oath"
        ]

    state["handling_options"] = segments
    state["current_step"] = "argument_generation"
    print("Options generated: ", segments)
    return state


def argument_generator(state: GraphState) -> GraphState:
    """Second LLM: Generate support and attack arguments for each segment"""
    arguments = []

    print(f"Generating arguments for segments: {state['handling_options']}")

    for segment in state["handling_options"]:
        # Generate supporting arguments
# Generate supporting arguments (reasons it’s hearsay)
        support_prompt = f"""For the following text segment, generate 2 concise reasons
        explaining why it qualifies as hearsay under Federal Rules of Evidence Rule 801(c):

        Segment:
        \"\"\"{segment}\"\"\"

        Format each reason on a new line starting with "Support:"."""
        support_response = call_llm(support_prompt, temperature=0.8)

        # Generate challenging arguments (reasons it might NOT be hearsay)
        attack_prompt = f"""For the following text segment, generate 2 concise reasons
        explaining why it might NOT be hearsay (e.g., it falls under a hearsay exception
        or is non‑hearsay):

        Segment:
        \"\"\"{segment}\"\"\"

        Format each reason on a new line starting with "Attack:"."""
        attack_response = call_llm(attack_prompt, temperature=0.8)

        # Parse arguments
        for line in support_response.split("\n"):
            if line.strip().startswith("Support:"):
                arg_content = line.replace("Support:", "").strip()
                if arg_content:
                    arguments.append(
                        Argument(
                            content=arg_content,
                            argument_type="support",
                            parent_option=segment,  # This should match exactly
                        )
                    )

        for line in attack_response.split("\n"):
            if line.strip().startswith("Attack:"):
                arg_content = line.replace("Attack:", "").strip()
                if arg_content:
                    arguments.append(
                        Argument(
                            content=arg_content,
                            argument_type="attack",
                            parent_option=segment,  # This should match exactly
                        )
                    )

    print(f"Generated {len(arguments)} arguments")
    state["arguments"] = arguments
    state["current_step"] = "human_review"
    # Don't override human_review_complete if it's already set
    if "human_review_complete" not in state:
        state["human_review_complete"] = False
    return state


def human_review(state: GraphState) -> GraphState:
    """Human-in-the-loop: Placeholder for Gradio interface"""
    # This node doesn't modify the state
    # The actual review happens in the Gradio interface
    return state


# Gradio Interface Manager
class CarePlanGradioInterface:
    def __init__(self):
        self.graph = None
        self.current_state = None
        self.current_thread_id = None
        self.app = None
        self.review_complete = False
        self.interface = None

    def format_arguments_display(self, state):
        """Format arguments for display in chat"""
        display_text = "## Generated Arguments for Review\n\n"

        for i, option in enumerate(state["handling_options"]):
            display_text += f"### Option {i+1}: {option}\n\n"
            option_args = [
                arg for arg in state["arguments"] if arg.parent_option == option
            ]

            # Group by type
            support_args = [
                arg for arg in option_args if arg.argument_type == "support"
            ]
            attack_args = [arg for arg in option_args if arg.argument_type == "attack"]

            if support_args:
                display_text += "**Supporting Arguments:**\n"
                for _, arg in enumerate(support_args):
                    arg_idx = state["arguments"].index(arg)
                    display_text += f"- [{arg_idx}] {arg.content}\n"
                display_text += "\n"

            if attack_args:
                display_text += "**Attacking Arguments (Concerns/Challenges):**\n"
                for _, arg in enumerate(attack_args):
                    arg_idx = state["arguments"].index(arg)
                    display_text += f"- [{arg_idx}] {arg.content}\n"
                display_text += "\n"

        display_text += "\n---\n"
        display_text += "**Available Actions:**\n"
        display_text += "- Type 'accept' to accept all arguments and continue\n"
        display_text += (
            "- Type 'remove [index]' to remove an argument (e.g., 'remove 3')\n"
        )
        display_text += "- Type 'add support [option_number] [argument]' to add a supporting argument\n"
        display_text += "- Type 'add attack [option_number] [argument]' to add an attacking argument\n"

        return display_text

    def process_user_input(self, message, history):
        """Process user commands during review"""
        if not self.current_state:
            return (
                "No active review session. Please start a care plan generation first."
            )

        message = message.strip().lower()

        if message == "accept":
            # Mark review as complete in the current state
            self.current_state["human_review_complete"] = True
            self.current_state["current_step"] = "argument_validation"
            self.review_complete = True
            return "✅ Arguments accepted. Proceeding to validation..."

        elif message.startswith("remove "):
            try:
                idx = int(message.split()[1])
                if 0 <= idx < len(self.current_state["arguments"]):
                    removed_arg = self.current_state["arguments"].pop(idx)
                    response = f"❌ Removed argument: {removed_arg.content}\n\n"
                    response += self.format_arguments_display(self.current_state)
                    return response
                else:
                    return "Invalid argument index. Please check the numbers in square brackets."
            except:
                return "Invalid remove command. Use format: 'remove [index]'"

        elif message.startswith("add support ") or message.startswith("add attack "):
            try:
                parts = message.split(maxsplit=3)
                arg_type = parts[1]  # 'support' or 'attack'
                option_idx = int(parts[2]) - 1
                arg_content = parts[3]

                if 0 <= option_idx < len(self.current_state["handling_options"]):
                    new_arg = Argument(
                        content=arg_content,
                        argument_type=arg_type,
                        parent_option=self.current_state["handling_options"][
                            option_idx
                        ],
                    )
                    self.current_state["arguments"].append(new_arg)
                    response = f"✅ Added {arg_type} argument: {arg_content}\n\n"
                    response += self.format_arguments_display(self.current_state)
                    return response
                else:
                    return f"Invalid option number. Please use 1-{len(self.current_state['handling_options'])}"
            except:
                return "Invalid add command. Use format: 'add [support/attack] [option_number] [argument text]'"

        else:
            return "Invalid command. Please use 'accept', 'remove [index]', or 'add [support/attack] [option] [text]'"

    def create_interface(self):
        """Create the Gradio interface"""
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .markdown-text {
            line-height: 1.6;
        }
        """

        with gr.Blocks(
            title="Legal Argumentation System", css=custom_css
        ) as self.interface:
            gr.Markdown("# ⚖️ Hearsay Snippet Analyzer")
            gr.Markdown(
            """
            This AI-driven system detects potential hearsay in text snippets under the Federal Rules of Evidence.
            The system will:
            1. Identify and flag segments that may constitute hearsay
            2. Generate supporting reasons why each segment qualifies as hearsay and possible counter-reasons
            3. Allow you to review, edit, or override each flag and justification
            4. Output the final annotated snippet with explanatory notes for every flagged segment
            """
            )

            with gr.Tab("🔍 Analyze Snippet"):
                patient_input = gr.Textbox(
                    label="Legal Text Snippet",
                    placeholder="Paste or upload a single legal text snippet to check for hearsay",
                    lines=5,
                )

                generate_btn = gr.Button("Detect Hearsay", variant="primary")

                # Progress display
                progress_display = gr.Markdown("", visible=False)

                # Chat interface for human review
                chatbot = gr.Chatbot(label="Hearsay Analysis Chat", height=500)
                msg = gr.Textbox(
                    label="Your Input",
                    placeholder="Type 'accept' to approve all, 'remove [index]' to remove, or 'add [support/attack] [option_num] [text]' to add",
                    submit_btn=True,
                    stop_btn=True,
                )

                with gr.Row():
                    toggle_dark = gr.Button(value="Toggle Dark")
                    toggle_dark.click(
                        None,
                        js="""
                        () => {
                            document.body.classList.toggle('dark');
                        }
                        """,
                    )

                # Final results display
                results_display = gr.Markdown("", visible=False)

                def generate_care_plan_gradio(input_text):
                    """Wrapper to run care plan generation with Gradio"""
                    if not input_text.strip():
                        return (
                            gr.update(
                                visible=True,
                                value="❌ Please enter legal text snippet to check for hearsay",
                            ),
                            [],
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    # Reset interface state
                    self.review_complete = False
                    self.current_state = None
                    self.current_thread_id = None
                    chat_history = []

                    try:
                        # Initialize the graph (create fresh instance to avoid state contamination)
                        self.graph = create_care_plan_graph()

                        # Create a new thread ID for this session
                        self.current_thread_id = f"session_{uuid4().hex}"

                        # Initial state - ensure human_review_complete is False
                        initial_state = {
                            "input_text": input_text,
                            "handling_options": [],
                            "arguments": [],
                            "validated_arguments": [],
                            "revised_answer": {},
                            "human_feedback": None,
                            "current_step": "care_plan_generation",
                            "human_review_complete": False,  # MUST be False for Gradio
                            "user_action": None,
                        }

                        cfg = {"configurable": {"thread_id": self.current_thread_id}}

                        # Add initial message
                        chat_history.append(
                            (
                                None,
                                "🤖 Generating hearsay segments and arguments... Please wait.",
                            )
                        )

                        # Run until human review
                        reached_human_review = False
                        for event in self.graph.stream(initial_state, cfg):
                            print(f"Event during generation: {list(event.keys())}")
                            for node_name, node_state in event.items():
                                if node_name == "human_review":
                                    self.current_state = (
                                        node_state.copy()
                                    )  # Make a copy to avoid reference issues
                                    reached_human_review = True
                                    break
                            if reached_human_review:
                                break

                        if not reached_human_review or not self.current_state:
                            raise Exception("Failed to reach human review stage")

                        # Display arguments for review
                        review_text = self.format_arguments_display(self.current_state)
                        chat_history.append((None, review_text))

                        return (
                            gr.update(
                                visible=True, value="## Review Generated Arguments"
                            ),
                            chat_history,
                            gr.update(visible=True),
                            gr.update(visible=False),
                        )
                    except Exception as e:
                        error_msg = f"❌ Error generating care plan: {str(e)}"
                        import traceback

                        print(f"Error details: {traceback.format_exc()}")
                        return (
                            gr.update(visible=True, value=error_msg),
                            [(None, error_msg)],
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                def continue_after_review(history):
                    """Continue processing after human review is complete"""
                    if not self.review_complete or not self.current_state:
                        return history, gr.update(visible=False)

                    try:
                        # Create a new thread ID for the continuation
                        # This avoids any checkpointing issues
                        continuation_thread_id = f"continue_{uuid4().hex}"
                        cfg = {"configurable": {"thread_id": continuation_thread_id}}

                        # Add processing message
                        history.append(
                            (
                                None,
                                "⏳ Validating arguments and generating final care plan...",
                            )
                        )

                        # Make sure the state has human_review_complete = True
                        self.current_state["human_review_complete"] = True
                        self.current_state["current_step"] = "argument_validation"

                        # Create a new graph instance to avoid state contamination
                        continuation_graph = create_care_plan_graph()

                        # Start from argument_validation by invoking with the modified state
                        # We'll run the remaining nodes: argument_validation -> plan_revision
                        final_state = None
                        events_seen = []

                        for event in continuation_graph.stream(self.current_state, cfg):
                            print(
                                f"Processing event after review: {list(event.keys())}"
                            )
                            events_seen.extend(list(event.keys()))
                            for node_name, node_state in event.items():
                                if node_name == "plan_revision":
                                    final_state = node_state

                        print(f"Events processed: {events_seen}")

                        if (
                            final_state
                            and "revised_answer" in final_state
                            and final_state["revised_answer"]
                        ):
                            print(
                                "Final State keys:",
                                final_state.get("revised_answer", {}).keys(),
                            )
                            # Format final results
                            results = f"""
## Final Hearsay Report

**Decision Confidence:** {final_state['revised_answer']['decision_confidence']}

**Argument Summary:**
- Total Arguments: {final_state['revised_answer']['argument_summary']['total_arguments']}
- Supporting: {final_state['revised_answer']['argument_summary']['support_arguments']}
- Challenging: {final_state['revised_answer']['argument_summary']['attack_arguments']}
- Average Validity: {final_state['revised_answer']['argument_summary']['avg_validity']:.2f}

**Recommendations:**
{final_state['revised_answer']['recommendations']}
                            """
                            history.append((None, "✅ Care plan generation complete!"))
                            return history, gr.update(visible=True, value=results)
                        else:
                            history.append(
                                (
                                    None,
                                    "❌ Error: Unable to generate final care plan. State may be incomplete.",
                                )
                            )
                            print(f"Final state: {final_state}")
                            return history, gr.update(visible=False)

                    except Exception as e:
                        import traceback

                        print(
                            f"Error in continue_after_review: {traceback.format_exc()}"
                        )
                        history.append((None, f"❌ Error during processing: {str(e)}"))
                        return history, gr.update(visible=False)

                def respond(message, chat_history):
                    """Handle chat responses"""
                    response = self.process_user_input(message, chat_history)
                    chat_history.append((message, response))

                    # Check if review is complete and continue processing
                    if self.review_complete:
                        chat_history, results = continue_after_review(chat_history)
                        return "", chat_history, results

                    return "", chat_history, gr.update()

                # Event handlers
                generate_btn.click(
                    generate_care_plan_gradio,
                    inputs=[patient_input],
                    outputs=[progress_display, chatbot, msg, results_display],
                )

                msg.submit(
                    respond,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot, results_display],
                )

            with gr.Tab("📋 Example Legal Text"):
                gr.Markdown(
                    """
                ### Example Legal Text

                **Legal Text 1**
                ```
                On the issue of whether Carl had knowledge of Amy's intentions, Carl told the questioning attorney on redirect examination that he knew Amy's intentions.
                ```

                **Legal Text 2**
                ```
                Alex is being sued for breach of contract relating to a delay in the shipment of mangos. To prove that the shipment was delayed, a witness for the plaintiffs testifies that he heard Alex complain about not being able to deliver the mangos in time. 
                ```

                **Legal Text 3**
                ```
                To prove that the trademarks of restaurant A and restaurant B created confusion, the fact that a customer called one and placed an order believing it to actually be the other.
                ```
                """
                )

            with gr.Tab("ℹ️ How to Use"):
                gr.Markdown("""
            ## How to Use This System

            ### Step 1: Paste or Upload Snippet
            Provide a single passage of legal text to analyze:
            - Paste or type the exact snippet into the textbox
            - Include full sentences or clauses you want checked

            ### Step 2: Review AI‑Flagged Hearsay Segments
            The system will detect potential hearsay segments and generate justifications. Interact using these commands:
            **Available Commands:**
            - `accept` – Confirm all flags and justifications  
            - `remove [index]` – Remove a specific reason by its index  
            - `add support [segment_number] [text]` – Add a hearsay justification  
            - `add attack [segment_number] [text]` – Add a non‑hearsay or exception reason  

            **Examples:**
            - `remove 1`  
            - `add support 2 The statement was made outside of court testimony`  
            - `add attack 1 The speaker was testifying under oath`

            ### Step 3: Download Annotated Snippet
            After accepting your changes, the system will:
            - Produce the original snippet annotated with hearsay flags  
            - Include explanatory notes and confidence scores  
            - Allow you to download the final annotated text

            ### Tips for Best Results
            - Use precise, self‑contained snippets  
            - Review each justification carefully  
            - Iterate with edits to refine annotations  
            """)


        return self.interface

    def launch(self, share=False):
        """Launch the Gradio interface"""
        if not self.interface:
            self.create_interface()
        self.interface.launch(share=share)


def argument_validator(state: GraphState) -> GraphState:
    """Third LLM: Evaluate validity and relevance of arguments"""
    validated_arguments = []

    for arg in state["arguments"]:
        prompt = f"""You are an expert analyst evaluating the validity and relevance of arguments 
        for elderly care planning.
        
        Handling Option: {arg.parent_option}
        
        Argument ({arg.argument_type}): {arg.content}
        
        Please evaluate this argument based on:
        1. Factual accuracy
        2. Relevance to elderly care and aging-in-place
        3. Practical considerations
        4. Evidence-based reasoning
        
        Provide a validity score between 0 and 1, where:
        - 0 = completely invalid/irrelevant
        - 0.5 = moderately valid
        - 1 = highly valid and relevant
        
        Response format: "Validity Score: X.XX"
        Include a brief explanation."""

        response = call_llm(prompt, temperature=0.3, max_tokens=256)

        # Extract validity score
        validity_score = 0.5  # default
        try:
            if "Validity Score:" in response:
                score_text = response.split("Validity Score:")[1].split()[0]
                validity_score = float(score_text.strip())
                validity_score = max(0, min(1, validity_score))  # Clamp to [0,1]
        except:
            pass

        arg.validity_score = validity_score
        validated_arguments.append(arg)

    state["validated_arguments"] = validated_arguments
    state["current_step"] = "plan_revision"
    return state


def care_plan_reviser(state: GraphState) -> GraphState:
    """Fourth LLM: Revise care plan based on validated arguments"""
    # Organize arguments by option and type
    arguments_by_option = {}
    for option in state["handling_options"]:
        arguments_by_option[option] = {"support": [], "attack": []}

    # Match arguments to options (handle potential mismatches)
    for arg in state["validated_arguments"]:
        matched = False
        # First try exact match
        if arg.parent_option in arguments_by_option:
            arguments_by_option[arg.parent_option][arg.argument_type].append(arg)
            matched = True
        else:
            # Try fuzzy matching if exact match fails
            for option in state["handling_options"]:
                # Check if the parent_option is a substring or vice versa
                if (
                    arg.parent_option in option
                    or option in arg.parent_option
                    or arg.parent_option.lower() in option.lower()
                    or option.lower() in arg.parent_option.lower()
                ):
                    arguments_by_option[option][arg.argument_type].append(arg)
                    matched = True
                    print(f"Fuzzy matched '{arg.parent_option}' to '{option}'")
                    break

        if not matched:
            print(
                f"Warning: Could not match argument with parent_option '{arg.parent_option}' to any handling option"
            )

    # Create prompt with weighted arguments
    prompt = f"""You are a legal AI assistant specialized in hearsay detection. 
    Based on the validated reasons for each flagged segment, produce the final annotated text snippet. 
    Highlight each hearsay segment inline and include explanatory notes.

    Original Snippet:
    {state['input_text']}

    Flagged Segments with Validated Reasons:
    """

    for option in state["handling_options"]:
        prompt += f"\n\nOption: {option}"

        # Add supporting arguments
        support_args = arguments_by_option[option]["support"]
        if support_args:
            prompt += "\n  Supporting arguments:"
            for arg in sorted(
                support_args, key=lambda x: x.validity_score, reverse=True
            ):
                prompt += f"\n    - [{arg.validity_score:.2f}] {arg.content}"

        # Add attacking arguments
        attack_args = arguments_by_option[option]["attack"]
        if attack_args:
            prompt += "\n  Attacking arguments (Concerns/Challenges):"
            for arg in sorted(
                attack_args, key=lambda x: x.validity_score, reverse=True
            ):
                prompt += f"\n    - [{arg.validity_score:.2f}] {arg.content}"

    prompt += """
    Based on the validated reasons and their validity scores, provide:
    1. The final annotated snippet with inline hearsay flags (it is a hearsay or not) and explanatory notes.
    2. A summary for each flagged segment showing its validity score and whether it is hearsay or non‑hearsay.
    3. An overall confidence score for the hearsay analysis of this snippet.
    4. Recommended next steps to verify or address any potential hearsay (e.g., obtaining live testimony, checking sources, invoking exceptions).
    """

    response = call_llm(prompt, temperature=0.6, max_tokens=1536)

    # Calculate decision confidence based on argument strengths
    decision_confidence = calculate_decision_confidence(state["validated_arguments"])

    state["revised_answer"] = {
        "recommendations": response,
        "decision_confidence": decision_confidence,
        "argument_summary": summarize_arguments(state["validated_arguments"]),
    }

    print(f"Answer revised. Decision confidence: {decision_confidence}")
    return state


# Helper Functions
def calculate_decision_confidence(arguments: List[Argument]) -> float:
    """Calculate overall confidence based on argument validity scores"""
    if not arguments:
        return 0.5

    support_scores = [
        arg.validity_score for arg in arguments if arg.argument_type == "support"
    ]
    attack_scores = [
        arg.validity_score for arg in arguments if arg.argument_type == "attack"
    ]

    avg_support = sum(support_scores) / len(support_scores) if support_scores else 0.5
    avg_attack = sum(attack_scores) / len(attack_scores) if attack_scores else 0.5

    # Higher support and lower attack scores increase confidence
    confidence = (avg_support + (1 - avg_attack)) / 2
    return round(confidence, 2)


def summarize_arguments(arguments: List[Argument]) -> Dict:
    """Summarize argument statistics"""
    summary = {
        "total_arguments": len(arguments),
        "support_arguments": len(
            [a for a in arguments if a.argument_type == "support"]
        ),
        "attack_arguments": len([a for a in arguments if a.argument_type == "attack"]),
        "avg_validity": (
            sum(a.validity_score for a in arguments) / len(arguments)
            if arguments
            else 0
        ),
    }
    return summary


# Define routing logic - FIXED VERSION
def route_after_human_review(state: GraphState) -> str:
    """Route based on whether human review is complete"""
    if state.get("human_review_complete", False):
        return "argument_validation"
    else:
        return "human_review"


# Build the graph
def create_care_plan_graph():
    """Create and configure the LangGraph workflow"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("care_plan_generation", care_plan_generator)
    workflow.add_node("argument_generation", argument_generator)
    workflow.add_node("human_review", human_review)
    workflow.add_node("argument_validation", argument_validator)
    workflow.add_node("plan_revision", care_plan_reviser)

    # Add edges
    workflow.set_entry_point("care_plan_generation")
    workflow.add_edge("care_plan_generation", "argument_generation")
    workflow.add_edge("argument_generation", "human_review")

    # Conditional edge for human review
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"human_review": "human_review", "argument_validation": "argument_validation"},
    )

    workflow.add_edge("argument_validation", "plan_revision")
    workflow.add_edge("plan_revision", END)

    # Compile with memory for checkpointing
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph


# Main execution functions
def generate_elderly_care_plan_terminal(input_text: str):
    """Terminal-based function to generate an argumentative care plan"""
    # Initialize the graph
    graph = create_care_plan_graph()

    # Initial state
    initial_state = {
        "input_text": input_text,
        "handling_options": [],
        "arguments": [],
        "validated_arguments": [],
        "revised_answer": {},
        "human_feedback": None,
        "current_step": "care_plan_generation",
        "human_review_complete": True,  # Skip human review for terminal version
        "user_action": None,
    }

    # Run the workflow
    config = {
        "configurable": {"thread_id": "care_plan_terminal"},
        "recursion_limit": 1000,
    }
    final_state = graph.invoke(initial_state, config)

    # Display results
    print("\n=== FINAL CARE PLAN ===")
    print(
        f"\nDecision Confidence: {final_state['revised_answer']['decision_confidence']}"
    )
    print(f"\nArgument Summary: {final_state['revised_answer']['argument_summary']}")
    print(f"\nRecommendations:\n{final_state['revised_answer']['recommendations']}")

    return final_state


# Example usage
if __name__ == "__main__":
    # Ensure model is loaded
    print(f"Loading HuggingFace model: {HUGGINGFACE_MODEL_NAME} on {DEVICE}")
    print("This may take a moment on first run...")

    # Launch Gradio interface
    # Launch Gradio interface
    print("\n" + "=" * 50)
    print("Launching Hearsay Snippet Analyzer Interface")
    print("=" * 50)
    print("\nThe interface will open in your default browser.")
    print("If it doesn't open automatically, click the URL shown below.")
    print("\nIn the interface, you can:")
    print("1. Paste or upload a legal text snippet")
    print("2. Review and refine AI‑flagged hearsay segments interactively")
    print("3. Download the final annotated snippet with explanatory notes")
    print("\nPress Ctrl+C to stop the server.\n")

    interface = CarePlanGradioInterface()
    interface.launch(share=True)  # Set share=True to create a public link
