#!/usr/bin/env python3
"""Fifth setting, stage 2 (12 Nov 2026): score with the open reranker every (question, tool) pair the per-dataset pools
need: fused top 60 (RRF k=60) and production's pool (meaning 30 + words 30). Pairs already scored in the 11 Nov
(question alone) or 12 Nov (with instruction) runs, with the identical question text and tool text, are reused from those
runs' rr-scores.npz; only new pairs are scored. Same model, revision, processor template, bfloat16, sdpa, 1,024-token cut,
batching as toolret-full.py / toolret-inst.py.
  python rerank-perdataset.py --dir=<work> --model=<work>/rerank-model --tag=q|i --deadline=<unix>
reads <work>/<tag>-stage1-perdataset.json, <work>/<tag>-scores-prev.npz, <work>/tr-corpus.jsonl, <work>/<tag>-queries.jsonl
writes <work>/out-<tag>/rr-stage1-perdataset.json (copy) and <work>/out-<tag>/rr-scores.npz (all needed pairs) + rr-run.json"""
import json, os, sys, time, shutil, numpy as np
a = {x.split("=")[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if "=" in x}
D, TAG = a["dir"], a["tag"]; MODEL = a.get("model", os.path.join(D, "rerank-model")); MAXLEN, TOKB, DEPTH = 1024, 24576, 60
DEADLINE = float(a.get("deadline", "0")); O = os.path.join(D, f"out-{TAG}"); os.makedirs(O, exist_ok=True)
S1 = json.load(open(os.path.join(D, f"{TAG}-stage1-perdataset.json"))); shutil.copy(os.path.join(D, f"{TAG}-stage1-perdataset.json"), os.path.join(O, "rr-stage1-perdataset.json"))
qr = [json.loads(l) for l in open(os.path.join(D, f"{TAG}-queries.jsonl"))]; ctext = [json.loads(l)["text"] for l in open(os.path.join(D, "tr-corpus.jsonl"))]
need = set()
for k, x in enumerate(S1):
    for i in x["fused"][:DEPTH] + x["union"]: need.add((k, i))
z = np.load(os.path.join(D, f"{TAG}-scores-prev.npz")); prev = {(int(q), int(d)): float(l) for q, d, l in zip(z["q"], z["d"], z["l"])}
SF = os.path.join(O, "rr-scores.npz")
done = {p: prev[p] for p in need if p in prev}; reused = len(done)
if os.path.exists(SF):
    z = np.load(SF); done.update({(int(q), int(d)): float(l) for q, d, l in zip(z["q"], z["d"], z["l"])})
todo = [p for p in need if p not in done]
print(f"{TAG}: pairs needed {len(need):,}; reused from earlier run {reused:,}; to score {len(todo):,}", flush=True)
def save():
    ks = list(done.items()); np.savez(SF, q=np.array([k[0] for k, _ in ks], dtype=np.int32), d=np.array([k[1] for k, _ in ks], dtype=np.int32), l=np.array([v for _, v in ks], dtype=np.float32))
NCUT = 0; MEANTOK = 0.0; t0 = time.time()
if todo:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoProcessor
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, rerank_max_length=MAXLEN); tk = proc.tokenizer
    qlen = {k: len(tk(qr[k]["text"], add_special_tokens=False)["input_ids"]) for k in {p[0] for p in todo}}
    dlen = {i: len(tk(ctext[i], add_special_tokens=False)["input_ids"]) for i in {p[1] for p in todo}}
    L = lambda p: min(MAXLEN, qlen[p[0]] + dlen[p[1]] + 12); todo.sort(key=L)
    NCUT = sum(1 for p in todo if qlen[p[0]] + dlen[p[1]] + 12 > MAXLEN); MEANTOK = float(np.mean([L(p) for p in todo]))
    print(f"tokens (capped {MAXLEN}): mean {MEANTOK:.0f}, truncated pairs {NCUT:,} of {len(todo):,} ({100 * NCUT / len(todo):.2f}%)", flush=True)
    i, last = 0, time.time()
    while i < len(todo):
        j = i; mx = 0
        while j < len(todo) and (j - i + 1) * max(mx, L(todo[j])) <= TOKB and j - i < 512: mx = max(mx, L(todo[j])); j += 1
        j = max(j, i + 1); batch = todo[i:j]
        bd = proc.process_queries_documents_crossencoder([{"question": qr[q]["text"], "doc_text": ctext[d], "doc_image": ""} for q, d in batch])
        with torch.no_grad():
            lg = model(input_ids=bd["input_ids"].to("cuda"), attention_mask=bd["attention_mask"].to("cuda"), pixel_values=None, return_dict=True).logits.float().view(-1).cpu().numpy()
        for p, l in zip(batch, lg): done[p] = float(l)
        i = j
        if time.time() - last > 300: save(); last = time.time(); print(f"rerank {i:,}/{len(todo):,} in {time.time() - t0:.0f}s", flush=True)
        if DEADLINE and time.time() > DEADLINE: save(); print("DEADLINE reached, checkpointed", flush=True); sys.exit(3)
save()
info = {"tag": TAG, "pairs_needed": len(need), "pairs_reused": reused, "pairs_scored": len(todo), "rerank_seconds": round(time.time() - t0, 1), "maxlen": MAXLEN,
        "pairs_cut_of_scored": NCUT, "mean_tokens_capped_scored": round(MEANTOK, 1)}
try:
    import torch, transformers; info.update({"transformers": transformers.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
except Exception: pass
json.dump(info, open(os.path.join(O, "rr-run.json"), "w")); print("DONE", info, flush=True)
