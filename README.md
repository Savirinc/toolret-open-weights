# ToolRet on open weights: scripts and raw results

This repository lets anyone check the ToolRet scores Unfragile reported in November 2026. It holds the scripts
that produced them, the candidates and reranker scores for every question, and a script that recomputes every
figure from those files against the benchmark's own answer key. No GPU is needed to check them.

**Built with Llama.** The reranker is `nvidia/llama-nemotron-rerank-vl-1b-v2`, under the NVIDIA Open Model License
and the Llama 3.2 Community License.

## The scores

nDCG@10, from 0 to 100. The headline figure is the mean of ToolRet's three categories (web, code, customized), each
the mean of its subsets, as in the ToolRet paper.

| Setting | Category mean | MTEB mean (35 subsets) | Query-weighted category mean | All questions pooled |
|---|---|---|---|---|
| Question alone, all tools in one pool | **41.96** | 42.75 | 43.77 | 47.58 |
| Question alone, within each category | **47.52** | 47.00 | 50.35 | 51.19 |
| Question plus instruction, all tools in one pool | **55.33** | 53.65 | 56.17 | 57.73 |
| Question plus instruction, within each category | **61.13** | 58.48 | 61.49 | 60.81 |
| Question alone, within each source dataset | **66.53** | 68.29 | 61.11 | 64.32 |
| Question plus instruction, within each source dataset | **75.95** | 76.48 | 70.30 | 72.16 |

In the last two rows each question is searched only among the tools of its own source dataset (11 to 16,072 tools,
median 95), as in Tables 10 and 11 of the ToolRet paper, whose best figures are 74.14 with instructions and 62.05
without.

The reranker reads the fused top 60 candidates. With instructions it was also run on the top 100 (55.26 and 60.92);
that depth was fixed before the run and both are reported.

## Check them yourself

```
pip install numpy datasets
python rescore.py
```

`rescore.py` downloads the answer key from `mteb/ToolRetrieval` at revision `76d45e560059754e289ea202462865a585679619`,
reads the files in `raw/`, and prints every stage in every setting. It exits with an error if any of the six
headline figures does not match. It takes a few minutes on a laptop.

## The fine-tuning study

`rescore_trained.py` recomputes the paper's fine-tuning tables the same way. It scores four systems on the same
questions: both models as released; the embedder fine-tuned on ToolRet's training set; the reranker fine-tuned on
it; and both. The questions are the 6,559 test questions with no exact or near copy in the training set, split into
the three source datasets the training set draws on (in domain, 1,707 questions) and the other 32 (out of domain,
4,852). The lists that define those slices are in `slices/`.

Full pipeline, nDCG@10, question plus instruction over all tools:

| Slice | Neither fine-tuned | Embedder | Reranker | Both |
|---|---|---|---|---|
| Out of domain (32 subsets) | 54.14 | 54.03 | 57.06 | 56.72 |
| In domain (3 subsets) | 66.20 | 68.71 | 74.89 | 77.62 |

Across the six settings, fine-tuning the embedder changes the out-of-domain pipeline by between -1.6 and +0.05
points; fine-tuning the reranker raises it by 0.8 to 2.9 points out of domain and by 1.7 to 8.9 in domain. It is one
run of one recipe for each model.

The fine-tuned weights are not published or served. Their candidate lists and reranker scores are in
`raw/trained-embedder-*/`, `raw/trained-reranker-*/` and `raw/trained-both-*/`.

## What is here

| Path | What it is |
|---|---|
| `rescore.py` | Recomputes all scores from the raw files and the benchmark's answer key |
| `rescore_trained.py` | Recomputes the fine-tuning tables on the de-duplicated slices, for all four systems |
| `slices/` | The test questions removed for having a copy in the training set, and the unseen-tools list. Ids only |
| `raw/question-alone/`, `raw/with-instruction/`, `raw/per-source-*/` | For each run: the candidates every stage returned for every question (`rr-stage1-*.json.gz`), the reranker's raw score for every pair it read (`rr-scores.npz`), the run log and software versions |
| `order/` | The question ids and tool ids in the order the runs used. Ids only |
| `results/` | The per-subset result files, and `numbers.json`, the single table every published figure is taken from |
| `scripts/` | The scripts that ran the pipeline, as they were run |

The benchmark's questions, tool texts and answer key are **not** copied here, because their datasets state no
licence. The scripts and `rescore.py` read them from Hugging Face at pinned revisions.

## How the runs were made

- **Data:** `mteb/ToolRetrieval` at `76d45e5` (7,961 questions, 44,453 tools, 35 subsets). Tool categories from
  `mangopy/ToolRet-Tools` at `e06c38c`. Instructions from `mangopy/ToolRet-Queries` at `b8c76ad`.
- **Meaning:** `nvidia/Nemotron-3.5-Embed-1B-BF16` at `e24ca1e`, 512 tokens, exact cosine, top 100.
- **Words:** BM25 (`bm25s` 0.3.16) over the same tool text, top 100.
- **Fusion:** reciprocal rank, k = 60.
- **Reranker:** `nvidia/llama-nemotron-rerank-vl-1b-v2` at `b8a9987`, text only, pairs cut at 1,024 tokens
  (1.8% of pairs cut without instructions, 2.0% with).
- **With instructions:** every stage reads ToolRet's instruction, a space, then the question.
- **Software:** transformers 5.20.0, torch 2.9.1. One NVIDIA A10G (question alone) and one NVIDIA L4 (with
  instructions).
- **Nothing was trained** for the six scores above. Both models were used as released. The fine-tuning study is
  separate and is described in its own section.

## What these scores do and do not show

- The two models' makers do not disclose all of their training data, so we cannot say the models never saw data
  like the benchmark's.
- With instructions over all tools, systems trained on ToolRet's own data report higher scores than ours:
  MagicSelector 59.90 (arXiv 2607.17751) and Tool-DE's models 57.01 (as listed in the same paper), both on the
  category mean. Averaged over all questions pooled, the CallBus embedder with a reranker reports 59.36
  (`yasinyaman/callbus-emb-8b`, trained on ToolRet's training data) against our 57.73.
- The ToolRet paper says each instruction was written by a model that was shown the target tools ("a target-aware
  strategy", Appendix B.6), so the with-instruction settings give a system more than a plain question would.
- The ToolRet paper prints 7,615 questions and 43,215 tools. The public release, unchanged since February 2025, has
  7,961 and 44,453, and that is what was run. The paper's exact subset could not be identified. Removing the source
  sets that best explain the gap in questions (ToolLens, then appbench or gpt4tools) moves the four headline scores
  by between -1.2 and +0.6 points (41.96 to between 40.79 and 42.41; 47.52 to 46.71 and 47.95; 55.33 to 54.92 and
  55.88; 61.13 to 60.84 and 61.71), changes no comparison, and leaves 892 tools of the gap unexplained. So comparisons with the paper are with
  higher reported scores, not a confirmed like-for-like improvement.
- ToolRet is a benchmark. It is not the questions people and agents ask a live search engine.

## Credit

- ToolRet: Shi, Wang, Yan, Ren, Wang, Yin and Ren, "Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval
  for Large Language Models", arXiv 2503.01763.
- The embedder and the reranker are NVIDIA's, released as open weights.

## Licence

The code in this repository is under the MIT licence (`LICENSE`). The result files are released for checking and
reuse with attribution. The models and datasets keep their own licences.
