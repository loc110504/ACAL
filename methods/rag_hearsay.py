"""
RAG-Enhanced Legal Inference with Azure OpenAI Embeddings
"""
from api_call import gpt_generate, llama_generate, phi4_generate
import pandas as pd
from pydantic import ValidationError
from methods.rag import RAGModule
import os

# ---- Initialize RAG components ----
print("🔧 Initializing RAG system with Azure OpenAI embeddings...")

# Initialize RAG module
rag = RAGModule(persist_directory="./chroma_db")

# Collection name for legal documents
COLLECTION_NAME = "legal_hearsay_docs"

# Try to load existing collection
vectorstore = rag.load_collection(collection_name=COLLECTION_NAME)

# If collection doesn't exist, create it from documents
if vectorstore is None:
    print(f"⚠️  Collection '{COLLECTION_NAME}' not found. Creating new collection...")
    docs_path = "./legal_documents"
    
    vectorstore = rag.create_collection_from_docs(
        docs_path=docs_path,
        collection_name=COLLECTION_NAME
    )
    
    if vectorstore is None:
        print("❌ Failed to create collection. Exiting.")
        exit(1)

print(f"✓ RAG system ready!\n")

# ---- RAG Configuration ----
RAG_TOP_K = 3  # Number of relevant documents to retrieve
MIN_SIMILARITY_SCORE = 0.5  # Minimum similarity threshold (lower score = more similar for some embeddings)


def retrieve_context(query: str, top_k: int = RAG_TOP_K):
    """
    Retrieve relevant legal context from RAG system.
    
    Args:
        query: The legal question/text to find context for
        top_k: Number of top documents to retrieve
        
    Returns:
        Tuple of (formatted context string, list of evidence dicts)
    """
    # Query RAG system
    evidences = rag.query_rag(
        query=query,
        top_k=top_k,
        collection_name=COLLECTION_NAME
    )
    
    if not evidences:
        return "", []
    
    # Filter by similarity score if needed
    # Note: Depending on the distance metric, lower scores may mean higher similarity
    # For cosine similarity with Chroma, lower distance = more similar
    relevant_evidences = evidences  # You can add filtering here if needed
    
    if not relevant_evidences:
        return "", []
    
    # Format context
    context_parts = []
    for i, evidence in enumerate(relevant_evidences, 1):
        text = evidence["text"]
        score = evidence["score"]
        source = evidence.get("source", "unknown")
        chunk_id = evidence.get("chunk_id", "N/A")
        
        context_parts.append(
            f"[Legal Context {i} - Relevance Score: {score:.3f}]\n"
            f"Source: {source} (Chunk {chunk_id})\n"
            f"{text.strip()}\n"
        )
    
    formatted_context = "\n".join(context_parts)
    return formatted_context, relevant_evidences


def build_rag_prompt(text: str, context: str) -> str:
    """
    Build system prompt with RAG context for hearsay detection.
    
    Args:
        text: The hearsay question
        context: Retrieved context from RAG system
        
    Returns:
        Complete system prompt with context
    """
    # Base prompt with few-shot examples
    base_prompt = """
    You are a professional legal reasoning assistant.
    HEARSAY RULE: Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted.

    FEW-SHOT EXAMPLES:
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
    
    # Add RAG context if available
    if context:
        context_section = f"""
        RELEVANT LEGAL AUTHORITY & PRECEDENTS
        The following legal materials may provide additional guidance for analyzing this case:

        {context}
        """
        base_prompt = base_prompt + "\n" + context_section
    
    # Add the actual question
    question_section = f"""
    NOW ANALYZE THIS CASE:

    Q: {text} Is there hearsay?
    A:

    RESPONSE FORMAT - Answer ONLY in valid JSON:
    {{
        "answer": "Yes" or "No",
        "explanation": "Provide 2-3 sentences explaining your reasoning."
    }}
    """
    
    return base_prompt + question_section


# ---- Load test data ----
print("📂 Loading test data...")
df = pd.read_csv("data/hearsay/test.tsv", sep="\t")
print(f"✓ Loaded {len(df)} test samples\n")

# ---- Loop chạy LLM với RAG ----
results = []

print("="*70)
print("STARTING RAG-ENHANCED INFERENCE")
print("="*70)

for idx, row in df.iterrows():
    text = row["text"]
    gold_answer = row["answer"]
    slice_type = row["slice"]
    context, evidences = retrieve_context(query=text, top_k=RAG_TOP_K)
    
    # ---- AUGMENT: Build prompt with context ----
    system_prompt = build_rag_prompt(text=text, context=context)

    # ---- GENERATE: Call LLM with augmented prompt ----
    result = llama_generate(system_prompt=system_prompt)

    if result:
        is_correct = (result.answer.strip().lower() == gold_answer.strip().lower())
        match_symbol = "✅" if is_correct else "❌"
        
        results.append({
            "index": idx,
            "text": text,
            "gold_answer": gold_answer,
            "slice": slice_type,
            "llm_answer": result.answer,
            "llm_explanation": result.explanation,
        })
        print(f"Done sample {idx}")
    else:
        results.append({
            "index": idx,
            "text": text,
            "gold_answer": gold_answer,
            "slice": slice_type,
            "llm_answer": None,
            "llm_explanation": None,
        })
        print(f"\n❌ Failed to parse response from LLM")
out_df = pd.DataFrame(results)
out_path = "rag_enhanced_llama_hearsay.tsv"
out_df.to_csv(out_path, sep="\t", index=False)
