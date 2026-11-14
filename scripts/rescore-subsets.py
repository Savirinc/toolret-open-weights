"""Rescore the 11/12 Nov ToolRet runs from their raw files (stage-1 lists + reranker logits) on subsets of the release:
drop whole source datasets (their questions AND their tools), to test whether the paper's smaller counts change our scores.
No GPU: candidate lists lose the dropped tools; the reranker's logits for the remaining pairs are reused."""
import json, math, sys, os, numpy as np
R = sys.argv[1]; CATS = json.load(open(sys.argv[2]))
cr = [json.loads(l)["id"] for l in open(f"{R}/tr-corpus.jsonl")]
qr = [json.loads(l) for l in open(f"{R}/tr-queries.jsonl")]
qrels = json.load(open(f"{R}/tr-qrels.json"))
def prefix(i): return i.split("_")[0]
def load(tag):
    z = np.load(f"{R}/{tag}-scores.npz"); sc = {(int(a), int(b)): float(c) for a, b, c in zip(z["q"], z["d"], z["l"])}
    return {s: json.load(open(f"{R}/{tag}-stage1-{s}.json")) for s in ("merged", "percat")}, sc
def ndcg(lst, rel):
    dcg = sum(rel.get(cr[i], 0) / math.log2(k + 2) for k, i in enumerate(lst[:10]))
    ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]; idcg = sum(g / math.log2(k + 2) for k, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
def run(tag, depth, drop_tasks, drop_prefixes):
    S1, sc = load(tag); out = {}
    gone = {i for i, t in enumerate(cr) if prefix(t) in drop_prefixes}
    for s, lists in S1.items():
        per = {}; short = 0
        for k, x in enumerate(lists):
            q = qr[k]
            if q["task"] in drop_tasks: continue
            rel = qrels[q["task"]].get(q["id"], {})
            if not x["dense"]: v = (0.0, 0.0)
            else:
                f = [i for i in x["fused"] if i not in gone][:depth]
                cand = [i for i in f if (k, i) in sc]
                if len(cand) < len(f): short += 1
                v = (ndcg([i for i in x["dense"] if i not in gone], rel), ndcg(sorted(cand, key=lambda i: -sc[(k, i)]), rel))
            per.setdefault(q["task"], []).append(v)
        tasks = {t: (100 * np.mean([a for a, _ in v]), 100 * np.mean([b for _, b in v]), len(v), CATS[t]) for t, v in per.items()}
        res = {}
        for j, name in ((0, "meaning"), (1, "rerank")):
            mteb = np.mean([v[j] for v in tasks.values()])
            cm = [np.mean([v[j] for v in tasks.values() if v[3] == c]) for c in ("web", "code", "customized")]
            qw = [sum(v[j] * v[2] for v in tasks.values() if v[3] == c) / sum(v[2] for v in tasks.values() if v[3] == c) for c in ("web", "code", "customized")]
            pooled = sum(v[j] * v[2] for v in tasks.values()) / sum(v[2] for v in tasks.values())
            res[name] = {"mteb": round(mteb, 4), "task3": round(float(np.mean(cm)), 4), "qw3": round(float(np.mean(qw)), 4), "pooled": round(pooled, 4)}
        out[s] = {"questions": sum(v[2] for v in tasks.values()), "tools": len(cr) - len(gone), "tasks": len(tasks), "short_candidate_lists": short, **res}
    return out
VARIANTS = {"full release": (set(), set()),
            "minus toollens": ({"toollens"}, {"toolLens"}),
            "minus toollens + appbench": ({"toollens", "appbench"}, {"toolLens", "appbench"}),
            "minus toollens + gpt4tools": ({"toollens", "gpt4tools"}, {"toolLens", "gpt4tools"})}
result = {}
for tag, depth in (("q", 60), ("i", 60), ("i", 100)):
    for name, (dt, dp) in VARIANTS.items():
        r = run(tag, depth, dt, dp); result[f"{'query-only' if tag == 'q' else 'with-instruction'} top{depth} | {name}"] = r
        print(tag, depth, name, json.dumps(r), flush=True)
json.dump(result, open(sys.argv[3], "w"), indent=1)
