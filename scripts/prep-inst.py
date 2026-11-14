#!/usr/bin/env python3
"""With-instruction ToolRet (12 Nov 2026): write <dir>/inst/tr-queries.jsonl, same ids and order as tr-queries.jsonl (MTEB),
text = ToolRet's instruction + " " + the MTEB query (instruction from mangopy/ToolRet-Queries@b8c76ad, toolret-instructions.json).
Links the stored tool vectors so toolret-embed.py embeds only the questions."""
import json, os, sys
D = sys.argv[1]; I = os.path.join(D, "inst"); os.makedirs(I, exist_ok=True)
ins = json.load(open(os.path.join(D, "toolret-instructions.json")))
rows = [json.loads(l) for l in open(os.path.join(D, "tr-queries.jsonl"))]
miss, diff, out = [], 0, []
for r in rows:
    x = ins.get(r["id"])
    if x is None: miss.append(r["id"]); out.append(dict(r)); continue
    if (x["query"] or "").strip() != (r["text"] or "").strip(): diff += 1
    out.append({**r, "text": f"{x['instruction'].strip()} {r['text'].strip()}".strip(), "instruction": x["instruction"]})
with open(os.path.join(I, "tr-queries.jsonl"), "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for n in ("tr-corpus.jsonl", "tr-corpus.f16.npy", "tr-corpus.ok.npy", "tr-corpus.meta.json"):
    p = os.path.join(I, n)
    if not os.path.exists(p): os.symlink(os.path.join(D, n), p)
print({"queries": len(rows), "missing_instruction": len(miss), "query_text_differs_from_toolret": diff}, flush=True)   # 12 Nov: no benchmark text in logs
if miss: sys.exit(f"missing instructions for {len(miss)} ids, e.g. {miss[:5]}")
