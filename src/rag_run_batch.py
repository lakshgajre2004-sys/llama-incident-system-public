# src/rag_run_batch.py
import json, os, time
from src.rag_demo import retrieve, build_prompt, get_embedder, get_embedder as _unused
# reuse generate_text in rag_demo or reimplement here
from src.rag_demo import generate_text

OUT = "results/rag_outputs.jsonl"
QUERIES = [
    # Example queries; replace/add realistic symptoms you extracted from logs or SRE runbooks
    "NameNode high latency and many replication warnings",
    "DataNode reports slow reads and Received block errors",
    "Frequent WARN messages about blockMap updates causing slowness",
    "Many short-lived connections failing across DataNodes",
    "IO exceptions during large file writes"
]

def save_result(res, out=OUT):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

def run_batch(queries=QUERIES, k=5):
    start = time.time()
    # clear old results
    if os.path.exists(OUT):
        os.remove(OUT)
    for q in queries:
        hits = retrieve(q, k=k)
        prompt = build_prompt(q, hits)
        gen = generate_text(prompt, max_len=300)
        res = {"query": q, "hits": hits, "rationale": gen, "timestamp": time.time()}
        save_result(res)
        print(f"[OK] Query done: {q}")
    print("Done. Time:", time.time()-start)

if __name__ == "__main__":
    run_batch()
