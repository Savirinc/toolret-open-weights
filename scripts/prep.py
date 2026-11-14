#!/usr/bin/env python3
"""Prep for toolret-full.py: check tool -> category coverage (tr-toolcats.json, built from mangopy/ToolRet-Tools@e06c38c),
and download the reranker snapshot at the pinned revision (then loaded offline from the local folder)."""
import json, os, sys
D = sys.argv[1]; REV = "b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a"
cr = [json.loads(l)["id"] for l in open(os.path.join(D, "tr-corpus.jsonl"))]; m = json.load(open(os.path.join(D, "tr-toolcats.json")))
from collections import Counter
print({"corpus": len(cr), "with_category": sum(1 for i in cr if i in m), "by_cat": Counter(m.get(i) for i in cr)}, flush=True)
from huggingface_hub import snapshot_download
p = snapshot_download("nvidia/llama-nemotron-rerank-vl-1b-v2", revision=REV, local_dir=os.path.join(D, "rerank-model"))
print("model ok", sorted(os.listdir(p)), flush=True)
