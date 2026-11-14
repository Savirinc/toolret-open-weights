#!/usr/bin/env python3
"""Recompute every published score from the raw files in this repository. No GPU and no model are needed.

    pip install numpy datasets
    python rescore.py

What it reads:
  raw/<run>/rr-stage1-<setting>.json.gz   the candidates each stage returned for every question (positions in order/corpus-order.txt)
  raw/<run>/rr-scores.npz                 the reranker's raw score for every (question, tool) pair it read
  order/query-order.jsonl                 the 7,961 question ids and their ToolRet subset, in the order the runs used
  order/corpus-order.txt                  the 44,453 tool ids, in the order the runs used
  scripts/toolret-categories.json         each subset's category (web, code, customized)
The answer key is NOT in this repository. It is downloaded from the dataset the runs used,
mteb/ToolRetrieval at revision 76d45e560059754e289ea202462865a585679619, so the scores are checked against the
benchmark's own labels and not against a copy of ours.

The score is nDCG@10 per question (gain = the label's relevance, discount log2(rank + 1), ideal from the labels),
averaged per subset, then averaged four ways: over the three categories with subset means inside each (the ToolRet
paper; the headline figure), over the 35 subsets (MTEB), over the three categories weighted by question count inside
each, and over all 7,961 questions pooled together.
"""
import gzip, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = "76d45e560059754e289ea202462865a585679619"
RUNS = {
    "question alone": ("raw/question-alone", {"meaning only": ("dense", None), "words only (BM25)": ("words", None),
                                               "fused": ("fused", None), "fused top 60 + reranker": ("fused", 60),
                                               "production pool + reranker": ("union", 0)}),
    "question plus instruction": ("raw/with-instruction", {"meaning only": ("dense", None), "words only (BM25)": ("words", None),
                                                            "fused": ("fused", None), "fused top 60 + reranker": ("fused", 60),
                                                            "fused top 100 + reranker": ("fused", 100),
                                                            "production pool + reranker": ("union", 0)}),
}
SETTINGS = {"merged": "all tools in one pool", "percat": "within each category"}
# The fifth setting: each question is searched only among the tools of its own source dataset.
PER_SOURCE = {"question alone": "raw/per-source-question-alone", "question plus instruction": "raw/per-source-with-instruction"}
PER_SOURCE_STAGES = {"meaning only": ("dense", None), "words only (BM25)": ("words", None), "fused": ("fused", None),
                     "fused top 60 + reranker": ("fused", 60), "production pool + reranker": ("union", 0)}


def load_labels(tasks):
    from datasets import load_dataset
    labels = {}
    for t in tasks:
        rows = load_dataset("mteb/ToolRetrieval", f"{t}-qrels", revision=REVISION)
        rows = rows[list(rows.keys())[0]]
        cols = rows.column_names
        qc = next(c for c in cols if "query" in c.lower()); dc = next(c for c in cols if "corpus" in c.lower() or "doc" in c.lower())
        sc = next(c for c in cols if "score" in c.lower() or "rel" in c.lower())
        m = labels.setdefault(t, {})
        for q, d, s in zip(rows[qc], rows[dc], rows[sc]):
            m.setdefault(q, {})[d] = int(s)
    return labels


def ndcg10(order, rel, ids):
    dcg = sum(rel.get(ids[i], 0) / math.log2(k + 2) for k, i in enumerate(order[:10]))
    ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]
    idcg = sum(g / math.log2(k + 2) for k, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    queries = [json.loads(l) for l in open(os.path.join(HERE, "order/query-order.jsonl"))]
    ids = [l.rstrip("\n") for l in open(os.path.join(HERE, "order/corpus-order.txt"))]
    cats = json.load(open(os.path.join(HERE, "scripts/toolret-categories.json")))
    tasks = sorted({q["task"] for q in queries})
    assert len(queries) == 7961 and len(ids) == 44453 and len(tasks) == 35, (len(queries), len(ids), len(tasks))
    print(f"downloading the answer key for {len(tasks)} subsets from mteb/ToolRetrieval@{REVISION[:7]} ...", flush=True)
    labels = load_labels(tasks)
    out = {}
    jobs = [(run, folder, stages, setting, name) for run, (folder, stages) in RUNS.items() for setting, name in SETTINGS.items()]
    jobs += [(run, folder, PER_SOURCE_STAGES, "perdataset", "within each source dataset") for run, folder in PER_SOURCE.items()]
    loaded = {}
    for run, folder, stages, setting, setting_name in jobs:
        if folder not in loaded:
            z = np.load(os.path.join(HERE, folder, "rr-scores.npz"))
            loaded = {folder: {(int(q), int(d)): float(l) for q, d, l in zip(z["q"], z["d"], z["l"])}}
        score = loaded[folder]
        if True:
            cand = json.load(gzip.open(os.path.join(HERE, folder, f"rr-stage1-{setting}.json.gz")))
            per = {name: {} for name in stages}
            for k, (q, c) in enumerate(zip(queries, cand)):
                rel = labels[q["task"]].get(q["id"], {})
                for name, (key, depth) in stages.items():
                    lst = c.get(key) or []
                    if depth is not None and lst:                      # reranked: order the pool by the reranker's score
                        pool = lst[:depth] if depth else lst
                        lst = sorted(pool, key=lambda i: -score[(k, i)])
                    per[name].setdefault(q["task"], []).append(ndcg10(lst, rel, ids) if lst else 0.0)
            for name, by_task in per.items():
                task_mean = {t: float(np.mean(v)) for t, v in by_task.items()}
                n = {t: len(v) for t, v in by_task.items()}
                mteb = 100 * sum(task_mean.values()) / 35
                paper = 100 * sum(float(np.mean([task_mean[t] for t in tasks if cats[t] == c])) for c in ("web", "code", "customized")) / 3
                weighted = 100 * sum(sum(task_mean[t] * n[t] for t in tasks if cats[t] == c) / sum(n[t] for t in tasks if cats[t] == c)
                                     for c in ("web", "code", "customized")) / 3
                pooled = 100 * sum(sum(v) for v in by_task.values()) / sum(n.values())
                out[(run, setting_name, name)] = (round(paper, 2), round(mteb, 2), round(weighted, 2), round(pooled, 2))
    print(f"\n{'run':27} {'setting':27} {'stage':30} {'paper mean':>10} {'MTEB mean':>10} {'weighted':>9} {'pooled':>7}")
    for (run, setting, name), (p, m, w, po) in out.items():
        print(f"{run:27} {setting:27} {name:30} {p:10.2f} {m:10.2f} {w:9.2f} {po:7.2f}")
    published = {("question alone", "all tools in one pool"): 41.96, ("question alone", "within each category"): 47.52,
                 ("question plus instruction", "all tools in one pool"): 55.33, ("question plus instruction", "within each category"): 61.13,
                 ("question alone", "within each source dataset"): 66.53, ("question plus instruction", "within each source dataset"): 75.95}
    ok = True
    print()
    for (run, setting), want in published.items():
        got = out[(run, setting, "fused top 60 + reranker")][0]
        same = abs(got - want) < 0.005
        ok &= same
        print(f"{'MATCH' if same else 'DIFFERS'}: {run}, {setting}: published {want:.2f}, recomputed {got:.2f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
