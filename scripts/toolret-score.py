#!/usr/bin/env python3
"""Score ToolRet (MTEB ToolRetrieval, query only) from the embeddings toolret-embed.py wrote: exact cosine over the merged
44,453-tool corpus for every question, nDCG@10 per question (trec_eval/pytrec_eval ndcg_cut_10: gain = qrel score,
discount log2(rank+1), ideal from the qrels), mean per subset; then the MTEB mean over 35 subsets and the paper's
three-category mean (plain mean of web, code, customized task means; mapping = ToolRet query metadata, the source means.py checks).
  python toolret-score.py --dir=./work"""
import json, os, sys, math, time
import numpy as np
a = {x.split("=")[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if "=" in x}
D = a.get("dir", "./work")
cr = [json.loads(l) for l in open(os.path.join(D, "tr-corpus.jsonl"))]; qr = [json.loads(l) for l in open(os.path.join(D, "tr-queries.jsonl"))]
C = np.load(os.path.join(D, "tr-corpus.f16.npy")).astype(np.float32); Q = np.load(os.path.join(D, "tr-queries.f16.npy")).astype(np.float32)
cok = np.load(os.path.join(D, "tr-corpus.ok.npy")); qok = np.load(os.path.join(D, "tr-queries.ok.npy"))
C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12); C[~cok] = 0
qrels = json.load(open(os.path.join(D, "tr-qrels.json"))); cats = json.load(open(os.path.join(D, "toolret-categories.json")))
cid = [r["id"] for r in cr]
per = {}; zero = []
t0 = time.time()
for s in range(0, len(qr), 512):
    blk = Q[s:s + 512]; blk = blk / np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-12)
    sc = blk @ C.T
    top = np.argpartition(-sc, 10, axis=1)[:, :10]
    for r in range(len(blk)):
        q = qr[s + r]; rel = qrels[q["task"]].get(q["id"], {})
        if not qok[s + r]:
            per.setdefault(q["task"], []).append(0.0); zero.append(q["id"]); continue
        t = top[r][np.argsort(-sc[r, top[r]])]
        dcg = sum(rel.get(cid[i], 0) / math.log2(k + 2) for k, i in enumerate(t))
        ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]
        idcg = sum(g / math.log2(k + 2) for k, g in enumerate(ideal))
        per.setdefault(q["task"], []).append(dcg / idcg if idcg else 0.0)
raw = {t: 100 * float(np.mean(v)) for t, v in per.items()}   # 12 Nov fix: means from unrounded per-task scores
tasks = {t: {"ndcg10": round(100 * float(np.mean(v)), 2), "n": len(v), "category": cats[t]} for t, v in sorted(per.items())}
assert len(tasks) == 35 and sum(x["n"] for x in tasks.values()) == len(qr)
mteb = round(sum(raw.values()) / 35, 2)
cmr = {c: float(np.mean([raw[t] for t, x in tasks.items() if x["category"] == c])) for c in ("web", "code", "customized")}
cm = {c: round(v, 2) for c, v in cmr.items()}
cq = {c: round(sum(raw[t] * x["n"] for t, x in tasks.items() if x["category"] == c) / sum(x["n"] for x in tasks.values() if x["category"] == c), 2) for c in cm}
out = {"benchmark": "MTEB ToolRetrieval (ToolRet, w/o instruction), dataset mteb/ToolRetrieval@76d45e560059754e289ea202462865a585679619",
       "corpus": len(cr), "queries": len(qr), "metric": "nDCG@10", "mteb_mean_35_subsets": mteb,
       "paper_three_category_mean": round(sum(cmr.values()) / 3, 2), "category_task_means": cm,
       "category_query_weighted_means": cq, "paper_three_category_mean_query_weighted": round(sum(cq.values()) / 3, 2),
       "category_task_counts": {c: sum(1 for x in tasks.values() if x["category"] == c) for c in cm},
       "scored_zero_not_embedded": zero, "tasks": tasks, "score_seconds": round(time.time() - t0, 1)}
for k in ("tr-corpus.meta.json", "tr-queries.meta.json", "tr-server.json"):
    p = os.path.join(D, k)
    if os.path.exists(p): out[k.replace(".json", "")] = json.load(open(p))
json.dump(out, open(os.path.join(D, "toolret-open-weights.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "tasks"}, indent=1)[:3000])
