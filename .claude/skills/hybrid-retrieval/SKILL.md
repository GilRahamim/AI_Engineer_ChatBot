---
name: hybrid-retrieval
description: Use this skill when implementing, modifying, or debugging the retrieval pipeline. Covers dense retrieval with ChromaDB, sparse retrieval with BM25, Reciprocal Rank Fusion, cross-encoder reranking, and query transformations like HyDE and multi-query. Triggers on tasks like "improve retrieval quality", "add reranker", "debug bad search results", "implement hybrid search", "add HyDE".
---

# Hybrid Retrieval Pipeline

This is the core engine. Quality of generation cannot exceed quality of retrieval — fix retrieval first.

## Pipeline order

```
query
  → (optional) query transform (HyDE / multi-query)
  → dense retrieval (ChromaDB)        ┐
  → sparse retrieval (BM25)           ├─ in parallel, top-20 each
                                      ┘
  → RRF fusion → top-20 merged
  → cross-encoder rerank → top-5
  → return chunks + metadata
```

Each stage is a clean function. Don't fuse stages "for performance" — modularity matters more than 50ms.

## Dense retrieval

Use ChromaDB with cosine similarity. Embed the query with **the same model** used at ingestion time (verify via `embedder_version`).

```python
results = collection.query(
    query_embeddings=[query_emb],
    n_results=20,
    where={"module": "Module 2"} if module_filter else None,  # metadata filter
    include=["documents", "metadatas", "distances"],
)
```

Cosine similarity:
$$\text{sim}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_d\|}$$

ChromaDB returns *distance* (1 - cosine), not similarity. Convert if you need scores: `score = 1 - distance`. For RRF you only need ranks, so don't bother converting.

## Sparse retrieval (BM25)

Use `bm25s` — orders of magnitude faster than `rank_bm25` and supports persistence.

```python
import bm25s

# at ingestion time:
retriever = bm25s.BM25()
retriever.index(bm25s.tokenize(corpus_texts, stopwords="en"))
retriever.save("data/bm25_index/")

# at query time:
results, scores = retriever.retrieve(bm25s.tokenize(query), k=20)
```

BM25 score:
$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

with $k_1 = 1.5$, $b = 0.75$ (defaults; rarely worth tuning for personal use).

**Why BM25 matters even if dense is great:** dense embeddings smooth over rare terms. A query like `"AdamW"` should hit chunks that contain *exactly* "AdamW", not chunks discussing optimizers in general. BM25 is precise on rare-token recall.

## RRF fusion

Reciprocal Rank Fusion is the simplest robust way to merge two ranked lists:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

with $k = 60$ (standard default — controls how much higher ranks dominate). Implementation:

```python
def reciprocal_rank_fusion(
    rankings: list[list[str]],   # each inner list is chunk_ids ordered by rank
    k: int = 60,
    top_n: int = 20,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in rankings:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
```

Why RRF over weighted score combination: no need to normalize scores between dense (cosine ∈ [-1, 1]) and sparse (BM25 ∈ [0, ∞)). Rank-based, model-agnostic.

## Cross-encoder reranking

This is the single biggest precision improvement.

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
pairs = [(query, chunk.text) for chunk in candidates]
scores = reranker.compute_score(pairs, normalize=True)
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:5]
```

Why it works: the bi-encoder used for indexing scores `sim(embed(q), embed(d))` — independent encoding. The cross-encoder scores `f(q, d)` — joint encoding, sees both at once, infinitely more discriminative.

It's slow (must encode every pair), so it only ever runs on top-20 from RRF, not on the full corpus.

GPU usage: ~600MB VRAM with fp16. Negligible alongside the LLM.

## Query transformations

### HyDE (Hypothetical Document Embeddings)

Ask the LLM to generate a hypothetical answer to the question; embed *that*; search with it. The hypothetical answer lives closer in vector space to real answer chunks than the question does.

```python
hyde_prompt = f"""Write a concise, technically detailed paragraph answering this question.
Don't ask follow-ups — just write the answer as if you knew it.
Question: {query}"""
hypothetical = llm.complete(hyde_prompt, max_tokens=200)
hyde_embedding = embedder.embed(hypothetical)
# now search with hyde_embedding instead of (or alongside) query embedding
```

Cost: one extra LLM call (~1-3 seconds locally). Skip for simple keyword queries; useful for conceptual ones.

### Multi-query expansion

Generate 3-5 paraphrases, retrieve for each, fuse all rankings via RRF.

```python
expansion_prompt = f"""Generate 4 different rephrasings of this question that someone studying for an AI engineering interview might ask. Output as a JSON list of strings, no other text.
Question: {query}"""
variants = json.loads(llm.complete(expansion_prompt))
all_rankings = [retrieve(v) for v in [query] + variants]
final = reciprocal_rank_fusion(all_rankings)
```

Both transformations are **opt-in** via config — they add latency. Default off, enable for hard queries or when measured to improve eval scores.

## Configuration

Everything tunable lives in `config/config.yaml`:

```yaml
retrieval:
  dense:
    top_k: 20
  sparse:
    top_k: 20
  fusion:
    k: 60
    top_n: 20
  reranker:
    enabled: true
    model: "BAAI/bge-reranker-v2-m3"
    top_n: 5
  transforms:
    hyde:
      enabled: false
      max_tokens: 200
    multi_query:
      enabled: false
      n_variants: 4
```

Never bake these into Python. All experiments must be reproducible from a config file.

## Debugging bad results

When retrieval fails for a query, check in this order:

1. **Is the answer even in the corpus?** Open `data/docs/`, grep for keywords. If it's not there, no retrieval can save you.
2. **Did sparse find it but dense miss it (or vice versa)?** Run dense and sparse separately, log top-20 from each. If only one branch found it, the other is dragging quality.
3. **Is the right chunk in top-20 but reranker buried it?** Log post-RRF top-20 *and* post-rerank top-5. If chunk #18 post-RRF disappears post-rerank, the reranker is misjudging.
4. **Is chunking the problem?** If the right chunk is split across two retrieved chunks neither of which is sufficient alone — fix chunking, not retrieval. See `document-ingestion` skill.

Log every stage's top results during eval. Without observability you're guessing.

## Common pitfalls

- **Querying with a different embedder than indexed with** — silent disaster. Guard at startup: load `embedder_version` from manifest, refuse to run if mismatched.
- **Forgetting metadata filters** — if the user asks "in module 3, what is X" and you don't filter by `module="Module 3"`, you waste retrieval budget on irrelevant chunks.
- **Reranking with a different language than the corpus** — `bge-reranker-v2-m3` is multilingual, but if you used a monolingual reranker, Hebrew queries against English corpus would fail. We use m3 specifically for this.
- **top-k too low** — `top_k=5` for dense + `top_k=5` for sparse leaves the reranker nothing to choose between. 20 + 20 → 20 → 5 is the sweet spot.
