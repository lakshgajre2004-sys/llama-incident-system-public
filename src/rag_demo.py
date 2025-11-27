# src/rag_demo.py
"""
RAG demo:
- retrieves top-k similar historical tickets using the FAISS index
- generates a short RCA + remediation using a small local HF model (gpt2 by default)
  *Replace the generator with a hosted API or a larger model if you have GPU and weights.*
Usage:
    python -m src.rag_demo
"""

import os
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Option A: local text-gen pipeline (transformers). Works without GPU but low-quality.
# Option B: use a hosted LLM API (OpenAI/HuggingFace) — swap generate_text() accordingly.
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

INDEX_DIR = "models/faiss"
INDEX_FILE = os.path.join(INDEX_DIR, "tickets.index")
META_FILE = os.path.join(INDEX_DIR, "mapping.csv")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError("FAISS index or mapping not found. Run src.build_faiss first.")
    index = faiss.read_index(INDEX_FILE)
    meta = pd.read_csv(META_FILE)
    return index, meta

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

def retrieve(query, k=TOP_K):
    s = get_embedder()
    qv = s.encode([query]).astype("float32")
    index, meta = load_index()
    D, I = index.search(qv, k)
    hits = []
    for idx in I[0]:
        if idx < 0:
            continue
        row = meta.iloc[idx]
        hits.append({
            "index": int(idx),
            "title": str(row.get("title","")),
            "description": str(row.get("description","")),
            "remediation": str(row.get("remediation",""))
        })
    return hits

def generate_text(prompt, max_len=200):
    """
    Default local generation (low-resource). Replace this with API call for better results.
    """
    if not HAS_TRANSFORMERS:
        return "Text-generation unavailable (transformers not installed). Install transformers or use an API."
    gen = pipeline("text-generation", model="gpt2", truncation=True)
    out = gen(prompt, max_length=max_len, do_sample=True, top_k=50, num_return_sequences=1)[0]['generated_text']
    return out

def build_prompt(query, hits):
    s = f"Context logs / query:\n{query}\n\nRelevant historical tickets:\n"
    for i,h in enumerate(hits, start=1):
        s += f"\n[{i}] Title: {h['title']}\nDesc: {h['description']}\nRemediation: {h['remediation']}\n"
    s += ("\nUsing the context above, produce a concise root-cause analysis (1-2 short paragraphs) "
          "and a short remediation checklist (3 bullet items). Keep it factual and avoid hallucination.")
    return s

def rag_step(query):
    hits = retrieve(query)
    prompt = build_prompt(query, hits)
    rationale = generate_text(prompt, max_len=300)
    return {
        "query": query,
        "hits": hits,
        "rationale": rationale
    }

def pretty_print(result):
    print("\n=== RAG DEMO RESULT ===\n")
    print("Query:\n", result["query"], "\n")
    print("Top hits:")
    for i,h in enumerate(result["hits"], start=1):
        print(f" {i}. {h['title'][:80]} --- {h['description'][:120]}")
    print("\nGenerated RCA + Remediation:\n")
    print(result["rationale"])
    print("\n=========================\n")

if __name__ == "__main__":
    # Example queries — replace with your real symptom
    examples = [
        "NameNode showing high latency and many replication warnings",
        "DataNode reporting 'Received block' errors and slow reads",
        "Frequent WARN messages about ledger or blockMap updates causing slowness"
    ]
    for q in examples:
        res = rag_step(q)
        pretty_print(res)
