#!/usr/bin/env python3
"""ToolRet WITH INSTRUCTION through Unfragile's search pipeline on OPEN WEIGHTS (12 Nov 2026; from toolret-full.py of 11 Nov).
Changes from 11 Nov, fixed before the run: every stage reads the question as ToolRet's instruction + " " + query
(inst/tr-queries.jsonl, built by prep-inst.py; the embedder vectors of that text are in inst/); the reranker reads the
FUSED TOP 100 (scored once; top 60 is read from the same scores) and the pool our live search uses at depth 60 (meaning 30 +
words 30) as before. Tools, tool vectors, BM25 text, tokenizer, RRF k=60, reranker and its 1,024-token cut: unchanged.
Stage 1 meaning: the stored open-embedder vectors (Nemotron-3.5-Embed-1B, toolret-embed.py), exact cosine, top 100.
Stage 1 words: BM25 (bm25s, our search service's tokenizer) over the same document text, top 100.
Fusion: reciprocal rank, k=60 (as in our search service), candidates = fused top 60.
  Also the pool our live search uses: meaning top 30 + words top 30 in their own order (our live search also has a trust
  signal, which has no role here).
Stage 2: open-weights reranker nvidia/llama-nemotron-rerank-vl-1b-v2 (transformers, trust_remote_code, local snapshot),
  text-only, prompt "question:{q} \\n \\n passage:{doc}" (the processor's v1 template), truncation at the END.
Settings: merged (all 44,453 tools; MTEB) and percat (each question searched in its own category's tools; 31 Oct).
  python toolret-full.py --dir=./work --model=./work/rerank-model --maxlen=1024
Resumable: rerank logits checkpoint to rr-scores.npz."""
import json, os, sys, math, re, time
import numpy as np
a = {x.split("=")[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if "=" in x}
D = a.get("dir", "./work"); QD = a.get("qdir", os.path.join(D, "inst")); MODEL = a.get("model", os.path.join(D, "rerank-model"))
MAXLEN = int(a.get("maxlen", 1024)); TOKB = int(a.get("tokbudget", 24576)); N2 = int(a.get("stage2", 100)); UNION_N = 60
SETTINGS = a.get("settings", "merged,percat").split(","); ONLY_STAGE1 = a.get("stage1only") == "1"
DEADLINE = float(a.get("deadline", "0"))   # unix time; stop reranking (checkpointed) after it
import bm25s
def tok(s): return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 1]   # the same tokenizer as our search service
cr = [json.loads(l) for l in open(os.path.join(D, "tr-corpus.jsonl"))]; qr = [json.loads(l) for l in open(os.path.join(QD, "tr-queries.jsonl"))]
qrels = json.load(open(os.path.join(D, "tr-qrels.json"))); cats = json.load(open(os.path.join(D, "toolret-categories.json")))
C = np.load(os.path.join(D, "tr-corpus.f16.npy")).astype(np.float32); Q = np.load(os.path.join(QD, "tr-queries.f16.npy")).astype(np.float32)
cok = np.load(os.path.join(D, "tr-corpus.ok.npy")); qok = np.load(os.path.join(QD, "tr-queries.ok.npy"))
C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12); C[~cok] = 0
Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
cid = [r["id"] for r in cr]; ctext = [r["text"] for r in cr]
toolcat = json.load(open(os.path.join(D, "tr-toolcats.json"))) if "percat" in SETTINGS else {}
print(f"corpus {len(cr):,} queries {len(qr):,}", flush=True)

def stage1(setting):
    f = os.path.join(QD, f"rr-stage1-{setting}.json")
    if os.path.exists(f): return json.load(open(f))
    pools = {"all": np.arange(len(cr))} if setting == "merged" else {c: np.array([i for i, x in enumerate(cid) if toolcat.get(x) == c]) for c in ("web", "code", "customized")}
    out = [None] * len(qr); t0 = time.time()
    for pname, pool in pools.items():
        bm = bm25s.BM25(); bm.index([tok(ctext[i]) or ["empty"] for i in pool], show_progress=False)
        Cp = C[pool]
        qidx = [k for k, q in enumerate(qr) if setting == "merged" or cats[q["task"]] == pname]
        for s in range(0, len(qidx), 256):
            blk = qidx[s:s + 256]; sc = Q[blk] @ Cp.T
            for r, k in enumerate(blk):
                q = qr[k]
                if not q["text"].strip() or not qok[k]: out[k] = {"dense": [], "words": [], "fused": [], "union": []}; continue
                top = np.argpartition(-sc[r], 100)[:100]; dense = [int(pool[i]) for i in top[np.argsort(-sc[r, top])]]
                res, _ = bm.retrieve([tok(q["text"]) or ["tool"]], k=min(100, len(pool)), show_progress=False)
                words = [int(pool[int(i)]) for i in res[0]]
                rrf = {}
                for lst in (dense, words):
                    for rk, i in enumerate(lst): rrf[i] = rrf.get(i, 0) + 1 / (60 + rk)
                fused = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]
                seen, union = set(), []
                for i in dense[:UNION_N // 2] + words[:UNION_N // 2]:
                    if i not in seen: seen.add(i); union.append(i)
                out[k] = {"dense": dense, "words": words, "fused": fused, "union": union}
        print(f"stage1 {setting}/{pname}: pool {len(pool):,}, {len(qidx):,} queries, {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(f, "w")); return out

S1 = {s: stage1(s) for s in SETTINGS}
need = set()
for s in SETTINGS:
    for k, x in enumerate(S1[s]):
        for i in x["fused"][:N2] + x["union"]: need.add((k, i))
print(f"pairs to rerank (both pools, all settings): {len(need):,}", flush=True)
if ONLY_STAGE1: sys.exit(0)

# ── stage 2: the open reranker
import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor
SF = os.path.join(QD, "rr-scores.npz"); done = {}
if os.path.exists(SF):
    z = np.load(SF); done = {(int(q), int(d)): float(l) for q, d, l in zip(z["q"], z["d"], z["l"])}
todo = [p for p in need if p not in done]
print(f"already scored {len(done):,}; to score {len(todo):,}", flush=True)
if todo:
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, rerank_max_length=MAXLEN)
    print(f"reranker loaded in {time.time() - t0:.0f}s; transformers {__import__('transformers').__version__} torch {torch.__version__}", flush=True)
    tk = proc.tokenizer
    qlen = [len(tk(q["text"], add_special_tokens=False)["input_ids"]) for q in qr]
    dlen = {}
    for i in sorted({d for _, d in todo}): dlen[i] = len(tk(ctext[i], add_special_tokens=False)["input_ids"])
    L = lambda p: min(MAXLEN, qlen[p[0]] + dlen[p[1]] + 12)
    todo.sort(key=L)
    NCUT = sum(1 for p in todo if qlen[p[0]] + dlen[p[1]] + 12 > MAXLEN); MEANTOK = float(np.mean([L(p) for p in todo]))
    print(f"tokens (capped {MAXLEN}): mean {MEANTOK:.0f}, truncated pairs {NCUT:,} of {len(todo):,} ({100 * NCUT / len(todo):.2f}%)", flush=True)
    def save():
        ks = list(done.items()); np.savez(SF, q=np.array([k[0] for k, _ in ks], dtype=np.int32), d=np.array([k[1] for k, _ in ks], dtype=np.int32), l=np.array([v for _, v in ks], dtype=np.float32))
    i, last, t0, nb = 0, time.time(), time.time(), 0
    while i < len(todo):
        bs = max(1, TOKB // L(todo[min(len(todo) - 1, i + 1)]))
        j = i; mx = 0
        while j < len(todo) and (j - i + 1) * max(mx, L(todo[j])) <= TOKB and j - i < 512: mx = max(mx, L(todo[j])); j += 1
        j = max(j, i + 1); batch = todo[i:j]
        feats = [{"question": qr[q]["text"], "doc_text": ctext[d], "doc_image": ""} for q, d in batch]
        bd = proc.process_queries_documents_crossencoder(feats)
        with torch.no_grad():
            lg = model(input_ids=bd["input_ids"].to("cuda"), attention_mask=bd["attention_mask"].to("cuda"), pixel_values=None, return_dict=True).logits.float().view(-1).cpu().numpy()
        for p, l in zip(batch, lg): done[p] = float(l)
        i = j; nb += 1
        if time.time() - last > 300: save(); last = time.time(); print(f"rerank {i:,}/{len(todo):,} in {time.time() - t0:.0f}s ({i / (time.time() - t0):.0f} pairs/s)", flush=True)
        if DEADLINE and time.time() > DEADLINE: save(); print("DEADLINE reached, checkpointed", flush=True); sys.exit(3)
    save(); print(f"rerank done {len(todo):,} pairs in {time.time() - t0:.0f}s", flush=True)
    json.dump({"rerank_seconds": round(time.time() - t0, 1), "pairs": len(todo), "maxlen": MAXLEN, "pairs_cut": NCUT, "pairs_cut_share": round(NCUT / len(todo), 4), "mean_tokens_capped": round(MEANTOK, 1), "transformers": __import__('transformers').__version__, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0)}, open(os.path.join(QD, "rr-run.json"), "w"))

# ── score
def ndcg(lst, rel):
    dcg = sum(rel.get(cid[i], 0) / math.log2(k + 2) for k, i in enumerate(lst[:10]))
    ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]; idcg = sum(g / math.log2(k + 2) for k, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
result = {"setting": "with instruction (ToolRet instruction + ' ' + query, every stage)", "stage2": N2, "union_n": UNION_N, "maxlen": MAXLEN, "settings": {}}
for s in SETTINGS:
    per = {}
    for k, x in enumerate(S1[s]):
        q = qr[k]; rel = qrels[q["task"]].get(q["id"], {})
        if not x["dense"]: v = {m: 0.0 for m in ("dense", "bm25", "fused", "fused60_rerank", "fused100_rerank", "union60_rerank")}
        else:
            c1 = x["fused"][:N2]; c2 = x["union"]; c60 = x["fused"][:60]
            v = {"dense": ndcg(x["dense"], rel), "bm25": ndcg(x["words"], rel), "fused": ndcg(x["fused"], rel),
                 "fused60_rerank": ndcg(sorted(c60, key=lambda i: -done[(k, i)]), rel), "fused100_rerank": ndcg(sorted(c1, key=lambda i: -done[(k, i)]), rel),
                 "union60_rerank": ndcg(sorted(c2, key=lambda i: -done[(k, i)]), rel),
                 "oracle_fused_top100": ndcg(sorted(c1, key=lambda i: -max(rel.get(cid[i], 0), 0)), rel)}   # ceiling: the FUSED top 100 in perfect order
        per.setdefault(q["task"], []).append(v)
    ms = ["dense", "bm25", "fused", "fused60_rerank", "fused100_rerank", "union60_rerank"]
    # 12 Nov fix: every mean below is taken from UNROUNDED per-task scores (raw); only the reported numbers are rounded
    raw = {t: {**{m: 100 * float(np.mean([v[m] for v in vs])) for m in ms}, "oracle_fused_top100": 100 * float(np.mean([v.get("oracle_fused_top100", 0.0) for v in vs])), "n": len(vs), "category": cats[t]} for t, vs in sorted(per.items())}
    tasks = {t: {**{m: round(100 * float(np.mean([v[m] for v in vs])), 2) for m in ms}, "oracle_fused_top100": round(100 * float(np.mean([v.get("oracle_fused_top100", 0.0) for v in vs])), 2), "n": len(vs), "category": cats[t]} for t, vs in sorted(per.items())}
    assert len(tasks) == 35 and sum(t["n"] for t in tasks.values()) == len(qr)
    ms2 = ms + ["oracle_fused_top100"]
    mteb = {m: round(sum(t[m] for t in raw.values()) / 35, 2) for m in ms2}
    cmr = {m: {c: float(np.mean([t[m] for t in raw.values() if t["category"] == c])) for c in ("web", "code", "customized")} for m in ms2}
    cm = {m: {c: round(x, 2) for c, x in cmr[m].items()} for m in ms2}
    three = {m: round(sum(cmr[m].values()) / 3, 2) for m in ms2}
    cq = {m: round(sum(sum(t[m] * t["n"] for t in raw.values() if t["category"] == c) / sum(t["n"] for t in raw.values() if t["category"] == c) for c in ("web", "code", "customized")) / 3, 2) for m in ms2}
    result["settings"][s] = {"mteb_mean_35": mteb, "paper_three_category_mean": three, "category_task_means": cm, "three_category_query_weighted": cq, "tasks": tasks}
    print(s, "MTEB35", mteb, "\n  3cat", three, flush=True)
for k in ("rr-run.json",):
    p = os.path.join(QD, k)
    if os.path.exists(p): result[k] = json.load(open(p))
json.dump(result, open(os.path.join(QD, "toolret-open-inst.json"), "w"), indent=1); print("SCORED", flush=True)
