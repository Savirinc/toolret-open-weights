#!/usr/bin/env python3
"""ToolRet (MTEB ToolRetrieval, query only) embeddings from the open-weights server, once for all 35 subsets.
  python toolret-embed.py --dir=./work --url=http://localhost:8000
Reads tr-corpus.jsonl (44,453 tools, merged corpus) and tr-queries.jsonl (7,961 questions); writes float16 npy files
in input order. Length-sorted requests of 256 texts; the server buckets by length inside each request. Failures are loud."""
import json, os, sys, time, urllib.request, urllib.error
import numpy as np
a = {x.split("=")[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if "=" in x}
DIR, URL = a.get("dir", "./work"), a.get("url", "http://localhost:8000")
REQ = int(a.get("req", 256)); MODEL = "nvidia/Nemotron-3.5-Embed-1B-BF16"
def post(texts, kind):
    body = {"input": texts, "model": MODEL, "input_type": kind, "encoding_format": "float", "truncate": "END"}
    r = urllib.request.Request(URL + "/v1/embeddings", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1200) as resp: return [d["embedding"] for d in json.load(resp)["data"]]
def run(name, kind):
    rows = [json.loads(l) for l in open(os.path.join(DIR, f"tr-{name}.jsonl"))]
    out_f = os.path.join(DIR, f"tr-{name}.f16.npy")
    if os.path.exists(out_f): print(name, "exists, skipped", flush=True); return
    V = np.zeros((len(rows), 2048), dtype=np.float16); ok = np.zeros(len(rows), dtype=bool)
    order = sorted([i for i, r in enumerate(rows) if r["text"].strip()], key=lambda i: len(rows[i]["text"]))
    failed = []; t0 = time.time()
    for s in range(0, len(order), REQ):
        b = order[s:s + REQ]
        try: vs = post([rows[i]["text"] for i in b], kind)
        except urllib.error.HTTPError as e:
            print(f"{name}: request {s // REQ} refused {e.code} {e.read()[:200]!r}; retrying one by one", flush=True); vs = []
            for i in b:
                try: vs += post([rows[i]["text"]], kind)
                except Exception as e2: vs.append(None); failed.append({"id": rows[i]["id"], "err": str(e2)[:200]})
        for i, v in zip(b, vs):
            if v is not None: V[i] = np.asarray(v, dtype=np.float32); ok[i] = True
        if (s // REQ) % 20 == 0: print(f"{name}: {min(s + REQ, len(order)):,}/{len(order):,} in {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    np.save(out_f, V); np.save(os.path.join(DIR, f"tr-{name}.ok.npy"), ok)
    meta = {"name": name, "rows": len(rows), "embedded": int(ok.sum()), "empty_skipped": len(rows) - len(order), "failed": failed, "seconds": round(secs, 1)}
    json.dump(meta, open(os.path.join(DIR, f"tr-{name}.meta.json"), "w"), indent=1); print(meta if len(failed) < 20 else {**meta, "failed": len(failed)}, flush=True)
run("queries", "query"); run("corpus", "passage")
try:
    with urllib.request.urlopen(URL + "/v1/models", timeout=10) as r: json.dump(json.load(r), open(os.path.join(DIR, "tr-server.json"), "w"))
except Exception as e: print("models unreadable", e)
print("DONE", flush=True)
