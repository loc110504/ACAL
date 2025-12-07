from api_call import llm_generate, llama_generate, phi4_generate
import pandas as pd

# ---- Load data ----
df = pd.read_csv("data/hearsay/test.tsv", sep="\t")

# ---- Loop chạy LLM ----
results = []

for idx, row in df.iterrows():
    text = row["text"]
    gold_answer = row["answer"]
    slice_type = row["slice"]

    # Tạo test prompt cho từng sample
    role_prompt = """
    You are a professional legal reasoning assistant.  
    """

    test_prompt = f"""
    Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted.

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

    Answer ONLY in JSON:
    {{
    "answer": "Yes or No",
    "explanation": "2–3 sentences explaining why the evidence is or is not hearsay."
    }}
    """

    # Gọi LLM OPENAI
    # result = llm_generate(
    #     system=role_prompt,
    #     messages=[{"role": "user", "content": test_prompt}],
    #     model="Llama-3.3-70B-Instruct",
    #     temperature=0,
    #     max_tokens=512,
    #     json_mode=True,
    # )

    result = phi4_generate(role_prompt, test_prompt)

    # Lưu kết quả
    results.append({
        "index": idx,
        "text": text,
        "gold_answer": gold_answer,
        "slice": slice_type,
        "llm_output": result
    })

    print(f"Done sample {idx}\n")

# ---- (Optional) Ghi kết quả ra file ----
out_df = pd.DataFrame(results)
out_df.to_csv("standard_prompt_phi4_hearsay.tsv", sep="\t", index=False)

print("💾 Saved to standard_prompt_llama_hearsay.tsv")
