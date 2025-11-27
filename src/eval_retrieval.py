# src/eval_retrieval.py
import json, pandas as pd
from collections import defaultdict
# load mapping of tickets -> metadata
meta = pd.read_csv("models/faiss/mapping.csv")  # contains timestamp/title/description

# load rag outputs
lines = [json.loads(l) for l in open("results/rag_outputs.jsonl","r",encoding="utf-8")]
# Example: if you use query->ground_truth timestamps, provide mapping here:
# ground_truth = {"Query text": ["2008-11-10 17:00","2008-11-10 19:00"], ...}
# For now compute proxy: % hits with same date (yy-mm-dd) as query mentions
def extract_date_from_text(t):
    # naive: look for yyyy-mm-dd or yyyy
    return None

results = []
for r in lines:
    q = r["query"]
    hits = r["hits"]
    # simple proxy: fraction of hits having 'HDFS' in title or description (example)
    match_count = sum(1 for h in hits if "HDFS" in (h.get("description","") + h.get("title","")))
    precision_at_k = match_count / max(1,len(hits))
    results.append({"query": q, "precision_at_k": precision_at_k, "match_count": match_count})
df = pd.DataFrame(results)
df.to_csv("results/rag_retrieval_metrics.csv", index=False)
print(df)
