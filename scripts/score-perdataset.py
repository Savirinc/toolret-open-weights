#!/usr/bin/env python3
"""Fifth setting, scoring (no GPU): nDCG@10 per question from the per-dataset stage-1 lists and reranker logits, then the
per-subset means and, WITHOUT intermediate rounding: the paper's three-category mean (task means within each category),
the query-weighted three-category mean, all questions pooled, and the MTEB mean over 35 subsets.
  python score-perdataset.py <raw dir with tr-corpus.jsonl tr-queries.jsonl tr-qrels.json> <toolret-categories.json>
         <q out dir> <i out dir> <result.json>"""
import json, math, sys, numpy as np
R, CATSF, QO, IO, OUT = sys.argv[1:6]; CATS = json.load(open(CATSF))
cr = [json.loads(l)["id"] for l in open(f"{R}/tr-corpus.jsonl")]; qr = [json.loads(l) for l in open(f"{R}/tr-queries.jsonl")]; qrels = json.load(open(f"{R}/tr-qrels.json"))
def ndcg(lst, rel):
    dcg = sum(rel.get(cr[i], 0) / math.log2(k + 2) for k, i in enumerate(lst[:10])); ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]
    idcg = sum(g / math.log2(k + 2) for k, g in enumerate(ideal)); return dcg / idcg if idcg else 0.0
res = {"setting": "fifth setting: each question searched only in its own source dataset's tools (ToolRet paper App. D.3)", "depth": 60, "runs": {}}
for run, O in (("question_alone", QO), ("with_instruction", IO)):
    S1 = json.load(open(f"{O}/rr-stage1-perdataset.json")); z = np.load(f"{O}/rr-scores.npz"); sc = {(int(a), int(b)): float(c) for a, b, c in zip(z["q"], z["d"], z["l"])}
    stages = {"meaning_only": "dense", "bm25_only": "words", "fused_rrf60": "fused", "fused_top60_rerank": 60, "production_pool60_rerank": "u"}
    per = {}
    for k, x in enumerate(S1):
        q = qr[k]; rel = qrels[q["task"]].get(q["id"], {}); v = {}
        for m, d in stages.items():
            if not x["dense"]: v[m] = 0.0
            elif d in ("dense", "words", "fused"): v[m] = ndcg(x[d], rel)
            elif d == "u": v[m] = ndcg(sorted(x["union"], key=lambda i: -sc[(k, i)]), rel)
            else: v[m] = ndcg(sorted(x["fused"][:d], key=lambda i: -sc[(k, i)]), rel)
        per.setdefault(q["task"], []).append(v)
    out = {}
    for m in stages:
        tm = {t: 100 * float(np.mean([v[m] for v in vs])) for t, vs in per.items()}; n = {t: len(vs) for t, vs in per.items()}
        out[m] = {"task_mean_3cat": round(float(np.mean([np.mean([tm[t] for t in tm if CATS[t] == c]) for c in ("web", "code", "customized")])), 2),
                  "query_weighted_3cat": round(float(np.mean([sum(tm[t] * n[t] for t in tm if CATS[t] == c) / sum(n[t] for t in tm if CATS[t] == c) for c in ("web", "code", "customized")])), 2),
                  "pooled_all_questions": round(sum(tm[t] * n[t] for t in tm) / sum(n.values()), 2), "mteb_mean_35": round(float(np.mean(list(tm.values()))), 2),
                  "category_task_means": {c: round(float(np.mean([tm[t] for t in tm if CATS[t] == c])), 2) for c in ("web", "code", "customized")},
                  "per_subset": {t: round(tm[t], 2) for t in sorted(tm)}}
    res["runs"][run] = {"questions": sum(len(v) for v in per.values()), "stages": out}
    print(run, {m: (o["task_mean_3cat"], o["query_weighted_3cat"], o["pooled_all_questions"], o["mteb_mean_35"]) for m, o in out.items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
