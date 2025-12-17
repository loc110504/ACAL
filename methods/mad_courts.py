"""
Multi-Agent Debate with RAG-enhanced Legal Inference
Adapted from original Multiagent Debate code
(COURTS version)
"""

import pandas as pd
import random
import json
from methods.rag import RAGModule
from api_call import llama_generate, gpt_generate, phi4_generate, gemini_generate

# ---------------- CONFIG ----------------
AGENTS = 3
ROUNDS = 2
RAG_TOP_K = 3
COLLECTION_NAME = "legal_hearsay_docs"

# ---------------- INIT RAG ----------------
rag = RAGModule(persist_directory="./chroma_db")
vectorstore = rag.load_collection(collection_name=COLLECTION_NAME)
if vectorstore is None:
    raise RuntimeError("RAG collection not found")

# ---------------- RAG UTILS ----------------
def retrieve_context(query: str, top_k: int):
    evidences = rag.query_rag(
        query=query,
        top_k=top_k,
        collection_name=COLLECTION_NAME
    )
    if not evidences:
        return ""

    context_parts = []
    for i, e in enumerate(evidences, 1):
        context_parts.append(
            f"[Legal Context {i} | score={e['score']:.3f}]\n{e['text']}"
        )
    return "\n\n".join(context_parts)


# ---------------- PROMPTS (COURTS) ----------------
def build_base_prompt(text: str, context: str):
    prompt = """
You are a professional legal reasoning assistant.

TASK:
Decide whether the post should be labeled "Yes" for COURTS or "No" otherwise.

LABEL DEFINITION (COURTS = "Yes"):
The post is about logistics of interacting with the court system or with lawyers, including:
- court procedures, filings, deadlines, hearings, appeals, records
- hiring, managing, or communicating with a lawyer

DECISION RULE:
Answer "Yes" ONLY IF the post is primarily about court or lawyer interaction logistics.
Otherwise answer "No".

OUTPUT REQUIREMENTS:
Answer ONLY in valid JSON:
{
  "answer": "Yes" or "No",
  "explanation": "2-3 sentences"
}
"""

    if context:
        prompt += f"\nRELEVANT LEGAL AUTHORITY:\n{context}\n"

    prompt += f"\nCASE:\n{text}\nLabel:\n"
    return prompt


def build_debate_prompt(text, context, other_agents_answers):
    debate_section = "\n\nOTHER AGENTS' ANALYSIS:\n"
    for i, ans in enumerate(other_agents_answers, 1):
        debate_section += f"\nAgent {i} Answer:\n{json.dumps(ans, indent=2)}\n"

    debate_section += """
Using the reasoning from other agents:
- Identify any incorrect assumptions
- Correct your own reasoning if needed
- Produce your best final answer
"""

    return build_base_prompt(text, context) + debate_section


# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/learned_hands_courts/test.tsv", sep="\t")

results = []

# ---------------- MAIN LOOP ----------------
for idx, row in df.iterrows():
    text = row["text"]
    gold = row["answer"]

    context = retrieve_context(text, RAG_TOP_K)

    # Agent memories: list of list
    agent_histories = [[] for _ in range(AGENTS)]

    for round_idx in range(ROUNDS):
        for agent_id in range(AGENTS):

            if round_idx == 0:
                prompt = build_base_prompt(text, context)
            else:
                others = [
                    agent_histories[j][-1]
                    for j in range(AGENTS) if j != agent_id
                ]
                prompt = build_debate_prompt(text, context, others)

            result = gpt_generate(system_prompt=prompt)

            if result is None:
                agent_histories[agent_id].append({
                    "answer": None,
                    "explanation": None
                })
            else:
                agent_histories[agent_id].append({
                    "answer": result.answer,
                    "explanation": result.explanation
                })

    # -------- FINAL DECISION --------
    final_answers = [h[-1]["answer"] for h in agent_histories]
    majority = max(set(final_answers), key=final_answers.count)

    results.append({
        "index": idx,
        "text": text,
        "gold_answer": gold,
        "final_answer": majority,
        "agent_answers": agent_histories
    })

    print(f"Done {idx}")


# ---------------- SAVE ----------------
out_df = pd.DataFrame(results)
out_df.to_json("mad_gemini_courts.json", orient="records", indent=2)
