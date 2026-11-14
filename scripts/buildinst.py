"""Build tr-queries-inst.jsonl: for every MTEB ToolRetrieval question (same ids/order as tr-queries.jsonl), ToolRet's own
instruction from mangopy/ToolRet-Queries@b8c76ad, plus the text sent in the with-instruction run: instruction + " " + query."""
import json, sys, collections
from datasets import load_dataset
REV = "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445"
tasks = json.load(open(sys.argv[1])); out = {}
for t in tasks:
    for r in load_dataset("mangopy/ToolRet-Queries", t, revision=REV)["queries"]:
        out[r["id"]] = (r["query"], r["instruction"])
json.dump({k: {"query": v[0], "instruction": v[1]} for k, v in out.items()}, open("toolret-instructions.json", "w"))
print(len(out), sum(1 for v in out.values() if not (v[1] or "").strip()), "empty instructions")
