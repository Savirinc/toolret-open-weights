#!/usr/bin/env python3
"""Recompute the fine-tuning tables from the raw files in this repository. No GPU and no model are needed.

    pip install numpy datasets
    python rescore_trained.py

For each of six settings it scores four systems on the same questions:
  released          the released embedder and the released reranker   (raw/question-alone, raw/with-instruction, raw/per-source-*)
  trained embedder  the fine-tuned embedder and the released reranker (raw/trained-embedder-*)
  trained reranker  the released embedder and the fine-tuned reranker (raw/trained-reranker-*)
  trained both      the fine-tuned embedder and the fine-tuned reranker (raw/trained-both-*)
and reports the embedder alone ("dense") and the full pipeline (fused top 60, reranked), on three slices of the
de-duplicated test set:
  out of domain   the 32 subsets the training set does not draw on
  in domain       toolbench, toolace, apigen
  unseen tools    questions none of whose labelled tools appears among the training positives
and, for reference only, on the full test set of 7,961 questions with nothing removed (the trained models never saw
the overlapping training rows, which were dropped before training, but the in-domain subsets dominate that figure).
The slices are defined by the files in slices/: dedup-test-matched.json lists the 1,402 test questions with an exact
or near copy in the training set (removed everywhere); unseen-answer-tools.json lists the unseen-tools questions.

The answer key is downloaded from mteb/ToolRetrieval at revision 76d45e560059754e289ea202462865a585679619.
The score is nDCG@10; each slice figure is the mean of subset means within a category, averaged over the categories
present in the slice (the ToolRet paper's average). Nothing is rounded before the final figure.
"""
import gzip, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = "76d45e560059754e289ea202462865a585679619"
IN_DOMAIN = {"toolbench", "toolace", "apigen"}
SETTINGS = [  # (label, run folder suffix, stage-1 file)
    ("question alone, all tools", "question-alone", "rr-stage1-merged.json.gz"),
    ("question alone, per category", "question-alone", "rr-stage1-percat.json.gz"),
    ("question alone, per source dataset", "per-source-question-alone", "rr-stage1-perdataset.json.gz"),
    ("with instruction, all tools", "with-instruction", "rr-stage1-merged.json.gz"),
    ("with instruction, per category", "with-instruction", "rr-stage1-percat.json.gz"),
    ("with instruction, per source dataset", "per-source-with-instruction", "rr-stage1-perdataset.json.gz"),
]
SYSTEMS = [("released", "raw/"), ("trained embedder", "raw/trained-embedder-"),
           ("trained reranker", "raw/trained-reranker-"), ("trained both", "raw/trained-both-")]


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


def slice_mean(per_question, keep, queries, cats):
    by_task = {}
    for k, v in enumerate(per_question):
        if keep[k]:
            by_task.setdefault(queries[k]["task"], []).append(v)
    by_cat = {}
    for t, vs in by_task.items():
        by_cat.setdefault(cats[t], []).append(float(np.mean(vs)))
    return 100 * float(np.mean([float(np.mean(v)) for v in by_cat.values()]))


def main():
    queries = [json.loads(l) for l in open(os.path.join(HERE, "order/query-order.jsonl"))]
    ids = [l.rstrip("\n") for l in open(os.path.join(HERE, "order/corpus-order.txt"))]
    cats = json.load(open(os.path.join(HERE, "scripts/toolret-categories.json")))
    dropped = set(json.load(open(os.path.join(HERE, "slices/dedup-test-matched.json"))))
    unseen = set(json.load(open(os.path.join(HERE, "slices/unseen-answer-tools.json"))))
    kept = [q["id"] not in dropped for q in queries]
    slices = {
        "out of domain": [k and q["task"] not in IN_DOMAIN for k, q in zip(kept, queries)],
        "in domain": [k and q["task"] in IN_DOMAIN for k, q in zip(kept, queries)],
        "unseen tools": [k and q["id"] in unseen for k, q in zip(kept, queries)],
        "full test set (nothing removed; for reference only)": [True] * len(queries),
    }
    print("de-duplicated questions:", sum(kept), "|", ", ".join(f"{n}: {sum(v)}" for n, v in slices.items()), flush=True)
    tasks = sorted({q["task"] for q in queries})
    print(f"downloading the answer key for {len(tasks)} subsets from mteb/ToolRetrieval@{REVISION[:7]} ...", flush=True)
    labels = load_labels(tasks)
    scores = {}
    out = {}
    for label, suffix, stage1 in SETTINGS:
        for system, prefix in SYSTEMS:
            folder = prefix + suffix
            if folder not in scores:
                z = np.load(os.path.join(HERE, folder, "rr-scores.npz"))
                scores[folder] = {(int(q), int(d)): float(l) for q, d, l in zip(z["q"], z["d"], z["l"])}
            sc = scores[folder]
            cand = json.load(gzip.open(os.path.join(HERE, folder, stage1)))
            dense, pipe = [], []
            for k, (q, c) in enumerate(zip(queries, cand)):
                rel = labels[q["task"]].get(q["id"], {})
                d = c.get("dense") or []
                f = (c.get("fused") or [])[:60]
                dense.append(ndcg10(d, rel, ids) if d else 0.0)
                pipe.append(ndcg10(sorted(f, key=lambda i: -sc[(k, i)]), rel, ids) if f else 0.0)
            for name, keep in slices.items():
                out[(name, label, system)] = (slice_mean(dense, keep, queries, cats), slice_mean(pipe, keep, queries, cats))
    names = [n for n, _ in SYSTEMS]
    for name in slices:
        print(f"\n{name}: full pipeline (fused top 60, reranked)\n{'setting':38}" + "".join(f"{n:>18}" for n in names))
        for label, *_ in SETTINGS:
            print(f"{label:38}" + "".join(f"{out[(name, label, n)][1]:18.2f}" for n in names))
        print(f"{name}: embedder alone\n{'setting':38}" + "".join(f"{n:>18}" for n in names[:2]))
        for label, *_ in SETTINGS:
            print(f"{label:38}" + "".join(f"{out[(name, label, n)][0]:18.2f}" for n in names[:2]))
    json.dump({f"{a}|{b}|{c}": {"embedder_alone": round(v[0], 2), "pipeline": round(v[1], 2)} for (a, b, c), v in out.items()},
              open(os.path.join(HERE, "results", "trained-slices.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
