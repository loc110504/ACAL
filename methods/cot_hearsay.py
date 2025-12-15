from api_call import gpt_generate, llama_generate, phi4_generate, gemini_generate
import pandas as pd
from pydantic import ValidationError

# ---- Load data ----
df = pd.read_csv("data/hearsay/test.tsv", sep="\t")
# ---- Loop chạy LLM ----
results = []

for idx, row in df.iterrows():
    text = row["text"]
    gold_answer = row["answer"]
    slice_type = row["slice"]

    # ---- Tạo system prompt ----
    system_prompt = f"""
You are a professional legal-reasoning assistant specializing in the U.S. Federal Rules of Evidence (FRE).
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

Q: {text} Is there hearsay?
A:

### Answer ONLY in JSON:
{{
"answer": "Yes or No",
"explanation": "2–3 sentences explaining why the evidence is or is not hearsay."
}}
"""

    # ---- Gọi Llama API ----
    print(f"🔍 Processing sample {idx}...")
    result = gemini_generate(system_prompt=system_prompt)

    if result:
        results.append({
            "index": idx,
            "text": text,
            "gold_answer": gold_answer,
            "slice": slice_type,
            "llm_answer": result.answer,
            "llm_explanation": result.explanation
        })
        print(f"✅ Done sample {idx}")
    else:
        results.append({
            "index": idx,
            "text": text,
            "gold_answer": gold_answer,
            "slice": slice_type,
            "llm_answer": None,
            "llm_explanation": None
        })
        print(f"⚠️ Failed to parse sample {idx}")

    # Nếu muốn test nhanh, chỉ chạy 1 mẫu


# ---- Ghi kết quả ra file ----
out_df = pd.DataFrame(results)
out_path = "cot_prompt_gemini_hearsay.tsv"
out_df.to_csv(out_path, sep="\t", index=False)

print(f"\n💾 Saved results to {out_path}")
