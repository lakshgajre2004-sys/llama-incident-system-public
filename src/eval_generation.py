# src/eval_generation.py
import json
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
lines = [json.loads(l) for l in open("results/rag_outputs.jsonl","r",encoding="utf-8")]
# load references mapping {query: reference_text}
# Example:
references = {
    "NameNode high latency and many replication warnings": "Root cause: ... Remediation: ..."
}
hyps = []
refs = []
for r in lines:
    q=r["query"]
    if q in references:
        hyps.append(r["rationale"])
        refs.append(references[q])
if not hyps:
    print("No references found; skip automatic metrics.")
else:
    bleu = corpus_bleu(hyps, [refs])
    print("BLEU:", bleu.score)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = [scorer.score(refs[i], hyps[i]) for i in range(len(hyps))]
    # aggregate
    import numpy as np
    avg = {k: sum(d[k].fmeasure for d in scores)/len(scores) for k in scores[0]}
    print("Avg ROUGE F:", avg)
