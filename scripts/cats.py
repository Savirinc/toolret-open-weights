import json
from datasets import load_dataset
from huggingface_hub import HfApi
m = {}; sizes = {}
for c in ("web", "code", "customized"):
    ids = list(load_dataset("mangopy/ToolRet-Tools", c)["tools"]["id"]); sizes[c] = len(ids)
    for i in ids:
        if i in m and m[i] != c: print("dup across cats", i)
        m.setdefault(i, c)
print(sizes, len(m), HfApi().dataset_info("mangopy/ToolRet-Tools").sha)
json.dump(m, open("tr-toolcats.json", "w"))
