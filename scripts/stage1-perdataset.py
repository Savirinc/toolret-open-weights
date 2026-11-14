#!/usr/bin/env python3
"""Fifth setting (ToolRet paper App. D.3, Tables 10 and 11): each question searched only within the toolset of its own
source dataset. Pool = every corpus tool whose id carries that subset's source prefix (the id up to its last "_",
e.g. toolLens_tool, gpt4tools_tool); in the released data each subset's positives use exactly one such prefix.
Stage 1 exactly as in toolret-full.py / toolret-inst.py: meaning top 100 by exact cosine (stored vectors), BM25 top 100
(bm25s, same tokenizer, one index per pool), RRF k=60, production pool = meaning 30 + words 30. No scoring here.
  python stage1-perdataset.py <raw dir> q|i  -> <raw dir>/<q|i>-stage1-perdataset.json"""
import json, re, sys, collections, numpy as np, bm25s
R, tag = sys.argv[1], sys.argv[2]
def tok(s): return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 1]
cr = [json.loads(l) for l in open(f"{R}/tr-corpus.jsonl")]; cid = [r["id"] for r in cr]; ctext = [r["text"] for r in cr]
qr = [json.loads(l) for l in open(f"{R}/{'tr' if tag == 'q' else 'i'}-queries.jsonl")]
qrels = json.load(open(f"{R}/tr-qrels.json"))
C = np.load(f"{R}/tr-corpus.f16.npy").astype(np.float32); C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
Q = np.load(f"{R}/{tag}-queries.f16.npy").astype(np.float32); Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
qok = np.array([bool(q["text"].strip()) for q in qr])
pre = lambda i: i.rsplit("_", 1)[0]
by_pre = collections.defaultdict(list)
for k, i in enumerate(cid): by_pre[pre(i)].append(k)
task_pre = {}
for t, qs in qrels.items():
    ps = {pre(d) for q in qs.values() for d in q}
    assert len(ps) == 1, (t, ps); task_pre[t] = ps.pop()
out = [None] * len(qr); sizes = {}
for t, p in sorted(task_pre.items()):
    pool = np.array(by_pre[p]); sizes[t] = len(pool)
    bm = bm25s.BM25(); bm.index([tok(ctext[i]) or ["empty"] for i in pool], show_progress=False)
    idx = [k for k, q in enumerate(qr) if q["task"] == t]
    sc = Q[idx] @ C[pool].T
    for r, k in enumerate(idx):
        if not qok[k]: out[k] = {"dense": [], "words": [], "fused": [], "union": []}; continue
        n = min(100, len(pool)); top = np.argpartition(-sc[r], n - 1)[:n] if n < len(pool) else np.arange(len(pool))
        dense = [int(pool[i]) for i in top[np.argsort(-sc[r, top])]]
        res, _ = bm.retrieve([tok(qr[k]["text"]) or ["tool"]], k=n, show_progress=False); words = [int(pool[int(i)]) for i in res[0]]
        rrf = {}
        for lst in (dense, words):
            for rk, i in enumerate(lst): rrf[i] = rrf.get(i, 0) + 1 / (60 + rk)
        fused = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        seen, union = set(), []
        for i in dense[:30] + words[:30]:
            if i not in seen: seen.add(i); union.append(i)
        out[k] = {"dense": dense, "words": words, "fused": fused, "union": union}
json.dump(out, open(f"{R}/{tag}-stage1-perdataset.json", "w"))
json.dump({"pool_prefix": task_pre, "pool_size": sizes}, open(f"{R}/perdataset-pools.json", "w"), indent=1)
print(tag, "done; pool sizes", sorted(sizes.values())[:5], "...", max(sizes.values()))
