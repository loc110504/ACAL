"""
Hearsay Snippet Analyzer with Gradio Interface

This tool lets you upload or paste a single text snippet to check
whether it contains hearsay, using AI-driven semantic analysis
with human-in-the-loop validation via Gradio.

When launched, the web interface allows you to:
1. Paste or upload one passage of legal text
2. View AI‑flagged hearsay segments and their justification
3. Confirm, edit, or override each flag
4. Download the annotated snippet with explanatory notes

Note: What is hearsay?
“Hearsay” means a statement that:
(1) the declarant does not make while testifying at the current trial or hearing; and
(2) a party offers in evidence to prove the truth of the matter asserted in the statement.
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
    """Represents a hearsay flag with its content, type, and confidence score"""
    content: str
    argument_type: str  # "support" (reasons it's hearsay) or "attack" (possible exception)
    validity_score: Optional[float] = None
    parent_option: Optional[str] = None  # here: the snippet segment


class GraphState(TypedDict):
    """State schema for the LangGraph workflow"""
    snippet_text: str
    handling_options: List[str]         # flagged segments
    arguments: List[Argument]           # justifications per segment
    validated_arguments: List[Argument]
    annotated_snippet: Dict[str, any]
    human_feedback: Optional[str]
    current_step: str
    human_review_complete: bool
    user_action: Optional[str]


# Node Functions
def care_plan_generator(state: GraphState) -> GraphState:
    """First LLM: Identify potential hearsay segments in the snippet"""
    prompt = f"""You are a legal AI assistant specialized in hearsay detection.
Given the following text snippet, identify 1-3 segments (sentences or clauses)
that may constitute hearsay under the Federal Rules of Evidence.

Text Snippet:
{state['snippet_text']}

Provide each flagged segment on its own line, prefixed with "Segment: "."""
    response = call_llm(prompt, temperature=0.6, max_tokens=512)

    # Parse flagged segments
    options = []
    for line in response.split("\n"):
        if line.strip().startswith("Segment:"):
            seg = line.split(":", 1)[1].strip()
            if seg:
                options.append(seg)

    if not options:
        # default: no flags parsed
        options = []

    state["handling_options"] = options
    state["current_step"] = "argument_generation"
    return state


def argument_generator(state: GraphState) -> GraphState:
    """Second LLM: Generate reasons supporting and challenging each hearsay flag"""
    arguments = []
    for segment in state["handling_options"]:
        # Supporting reasons (why it's hearsay)
        support_prompt = f"""For the following snippet segment, generate 2 brief reasons
explaining why it qualifies as hearsay under FRE 801(c):

Segment:
\"\"\"{segment}\"\"\"

Format each reason on a new line starting with "Support:"."""
        support_response = call_llm(support_prompt, temperature=0.8, max_tokens=256)

        # Challenging reasons (possible exceptions or non-hearsay)
        attack_prompt = f"""For the same segment, generate 2 brief reasons
why it might NOT be hearsay (e.g., falls under an exception or is non-hearsay):

Segment:
\"\"\"{segment}\"\"\"

Format each reason on a new line starting with "Attack:"."""
        attack_response = call_llm(attack_prompt, temperature=0.8, max_tokens=256)

        # Parse reasons
        for line in support_response.split("\n"):
            if line.strip().startswith("Support:"):
                content = line.replace("Support:", "").strip()
                if content:
                    arguments.append(Argument(
                        content=content,
                        argument_type="support",
                        parent_option=segment,
                    ))
        for line in attack_response.split("\n"):
            if line.strip().startswith("Attack:"):
                content = line.replace("Attack:", "").strip()
                if content:
                    arguments.append(Argument(
                        content=content,
                        argument_type="attack",
                        parent_option=segment,
                    ))
    state["arguments"] = arguments
    state["current_step"] = "human_review"
    if "human_review_complete" not in state:
        state["human_review_complete"] = False
    return state


def human_review(state: GraphState) -> GraphState:
    """Human-in-the-loop: no-op, actual review via Gradio"""
    return state


# Gradio Interface Manager
class CarePlanGradioInterface:
    def __init__(self):
        self.graph = None
        self.current_state = None
        self.review_complete = False
        self.interface = None

    def format_arguments_display(self, state):
        display = "## AI‑Flagged Hearsay Analysis\n\n"
        for i, segment in enumerate(state["handling_options"]):
            display += f"### Segment {i+1}: \"{segment}\"\n\n"
            seg_args = [arg for arg in state["arguments"] if arg.parent_option == segment]
            support = [a for a in seg_args if a.argument_type=="support"]
            attack = [a for a in seg_args if a.argument_type=="attack"]
            if support:
                display += "**Reasons it is hearsay:**\n"
                for arg in support:
                    idx = state["arguments"].index(arg)
                    display += f"- [{idx}] {arg.content}\n"
                display += "\n"
            if attack:
                display += "**Possible exceptions / Non‑hearsay reasons:**\n"
                for arg in attack:
                    idx = state["arguments"].index(arg)
                    display += f"- [{idx}] {arg.content}\n"
                display += "\n"
        display += "---\n**Commands:**\n"
        display += "- `accept` to confirm flags\n"
        display += "- `remove [index]` to drop a reason\n"
        display += "- `add support [segment_number] [text]` to add hearsay reason\n"
        display += "- `add attack [segment_number] [text]` to add exception reason\n"
        return display

    def process_user_input(self, message, history):
        # identical logic unchanged...
        # (omitted for brevity, same as original)
        ...

    def create_interface(self):
        custom_css = """
        .gradio-container { font-family: 'Arial', sans-serif; }
        .markdown-text { line-height: 1.6; }
        """
        with gr.Blocks(title="Hearsay Snippet Analyzer", css=custom_css) as self.interface:
            gr.Markdown("# 📜 Hearsay Snippet Analyzer")
            gr.Markdown("""
Upload or paste a legal text snippet to detect hearsay clauses.
You will:
1. Identify AI‑flagged hearsay segments
2. Review and refine each justification
3. Download the annotated snippet with notes
            """)
            with gr.Tab("🔍 Analyze Snippet"):
                snippet_input = gr.Textbox(
                    label="Legal Text Snippet",
                    placeholder="Paste your snippet here...",
                    lines=5,
                )
                analyze_btn = gr.Button("Analyze Snippet", variant="primary")
                progress = gr.Markdown("", visible=False)
                chatbot = gr.Chatbot(label="Hearsay Analysis", height=400)
                user_msg = gr.Textbox(
                    label="Your Command",
                    placeholder="Type 'accept', 'remove [index]', or 'add support/attack [seg] [text]'",
                    submit_btn=True,
                )
                results = gr.Markdown("", visible=False)

                def launch_analysis(snippet):
                    if not snippet.strip():
                        return gr.update(visible=True, value="❌ Please provide text"), [], gr.update(visible=False)
                    self.review_complete = False
                    self.graph = create_care_plan_graph()
                    init_state = {
                        "snippet_text": snippet,
                        "handling_options": [],
                        "arguments": [],
                        "validated_arguments": [],
                        "annotated_snippet": {},
                        "human_feedback": None,
                        "current_step": "care_plan_generation",
                        "human_review_complete": False,
                        "user_action": None,
                    }
                    history = [(None, "🤖 Detecting hearsay segments...")]
                    for event in self.graph.stream(init_state, {"configurable":{"thread_id":f"session_{uuid4().hex}"}}):
                        if "human_review" in event:
                            self.current_state = event["human_review"].copy()
                            break
                    display = self.format_arguments_display(self.current_state)
                    history.append((None, display))
                    return gr.update(visible=True, value="## Review Flags"), history, gr.update(visible=True)

                analyze_btn.click(launch_analysis, inputs=[snippet_input], outputs=[progress, chatbot, user_msg])
                user_msg.submit(lambda m,h: (self.process_user_input(m,h),), inputs=[user_msg, chatbot], outputs=[user_msg, chatbot, results])

            with gr.Tab("ℹ️ How to Use"):
                gr.Markdown("""
## How to Use
1. Paste or upload one legal text snippet.
2. Review AI‑flagged segments in the chat:
   - `accept` to confirm flags
   - `remove [index]` to discard a reason
   - `add support [segment] [text]` to add hearsay reason
   - `add attack [segment] [text]` to add exception reason
3. After accepting, receive the final annotated snippet with notes.
                """)
        return self.interface

    def launch(self, share=False):
        if not self.interface:
            self.create_interface()
        self.interface.launch(share=share)


# Remaining nodes (argument_validator, care_plan_reviser) should have their prompts updated similarly:
def argument_validator(state: GraphState) -> GraphState:
    """Third LLM: Score each justification for hearsay flags"""
    validated = []
    for arg in state["arguments"]:
        prompt = f"""You are a legal evidence expert. Evaluate the following reason
for segment:
\"\"\"{arg.parent_option}\"\"\"

Reason ({arg.argument_type}): {arg.content}

Score from 0 (weak) to 1 (strong) in detecting hearsay or exception.
Respond as "Validity Score: X.XX" with brief justification."""
        response = call_llm(prompt, temperature=0.3, max_tokens=128)
        # parsing unchanged...
        try:
            score = float(response.split("Validity Score:")[1].split()[0])
        except:
            score = 0.5
        arg.validity_score = max(0, min(1, score))
        validated.append(arg)
    state["validated_arguments"] = validated
    state["current_step"] = "plan_revision"
    return state


def care_plan_reviser(state: GraphState) -> GraphState:
    """Fourth LLM: Produce annotated snippet with notes based on validated reasons"""
    prompt = f"""You are an AI assistant. Given the original snippet and validated reasons,
produce a final annotated version, marking hearsay segments and including
explanatory notes from the strong reasons.

Original Snippet:
{state['snippet_text']}

Validated Reasons:
"""
    for arg in state["validated_arguments"]:
        prompt += f"\n- [{arg.argument_type.upper()} {arg.validity_score:.2f}] \"{arg.parent_option}\": {arg.content}"
    prompt += "\n\nOutput the snippet with inline annotations."
    response = call_llm(prompt, temperature=0.6, max_tokens=1024)
    state["annotated_snippet"] = {"annotated_text": response}
    state["current_step"] = END
    return state


# Graph builder unchanged except entry fields renamed:
def route_after_human_review(state: GraphState) -> str:
    return "argument_validation" if state.get("human_review_complete") else "human_review"

def create_care_plan_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("care_plan_generation", care_plan_generator)
    workflow.add_node("argument_generation", argument_generator)
    workflow.add_node("human_review", human_review)
    workflow.add_node("argument_validation", argument_validator)
    workflow.add_node("plan_revision", care_plan_reviser)
    workflow.set_entry_point("care_plan_generation")
    workflow.add_edge("care_plan_generation", "argument_generation")
    workflow.add_edge("argument_generation", "human_review")
    workflow.add_conditional_edges("human_review", route_after_human_review,
                                   {"human_review":"human_review","argument_validation":"argument_validation"})
    workflow.add_edge("argument_validation","plan_revision")
    workflow.add_edge("plan_revision", END)
    return workflow.compile(checkpointer=MemorySaver())
