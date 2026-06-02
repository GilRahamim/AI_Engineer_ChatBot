---
name: rag-evaluation
description: Use this skill when measuring RAG system quality, building or extending the test set, running RAGAS evaluation, or comparing system variants. Covers metric selection, golden dataset construction, baseline tracking, and result interpretation. Triggers on tasks like "evaluate the system", "did this change help", "build test set", "run RAGAS", "compare retrieval strategies".
---

# RAG Evaluation

Without measurement you're guessing. Every architectural change — chunker, retriever, reranker, prompt — must be evaluated against a baseline before being committed.

## Metrics that matter

Four core metrics, all in `[0, 1]`:

### Faithfulness
**Question**: Does every claim in the answer follow from the retrieved context?
**Why it matters**: Catches hallucination. The single most important metric for trust.
**Failure mode it detects**: Model invents facts not in context.

### Answer relevance
**Question**: Does the answer address the question that was asked?
**Why it matters**: Catches off-topic responses, even when they're factually grounded.
**Failure mode it detects**: Model answers a related-but-different question.

### Context precision
**Question**: Of the retrieved chunks, how many are actually relevant to the answer?
**Why it matters**: Measures *retrieval*, not generation. High = retriever isn't dragging in noise.
**Failure mode it detects**: Top-k full of off-topic chunks; reranker not working.

### Context recall
**Question**: Of the chunks needed to answer fully, how many were retrieved?
**Why it matters**: The other side of precision. Catches missed-information failures.
**Failure mode it detects**: Right answer is in the corpus but the retriever missed it.

These four together cover both pipelines (retrieval and generation). Faithfulness + answer relevance grade the LLM; precision + recall grade the retriever.

## RAGAS setup

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# RAGAS needs a Dataset with these columns:
# - question: str
# - answer: str (what your system produced)
# - contexts: list[str] (the chunks your retriever returned)
# - ground_truth: str (the expected answer)

ds = Dataset.from_list([
    {
        "question": "What is the vanishing gradient problem?",
        "answer": system_output,
        "contexts": [c.text for c in retrieved_chunks],
        "ground_truth": golden_answer,
    },
    ...
])

results = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
```

RAGAS internally calls an LLM to judge — by default OpenAI. **Configure it to use your local LM Studio** so eval is reproducible and free:

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

eval_llm = LangchainLLMWrapper(ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model",
))
results = evaluate(ds, metrics=[...], llm=eval_llm)
```

Caveat: a local 20B model judging itself is weaker than GPT-4 judging it. Numbers will be noisier. Use deltas (`v2 - v1`), not absolutes — they're more reliable than the raw scores.

## Building the golden test set

Store at `data/eval/golden_qa.jsonl`. One JSON object per line:

```json
{"id": "q001", "question": "What is the vanishing gradient problem?", "ground_truth": "...", "module": "Module 2", "difficulty": "easy"}
```

### Construction rules

1. **20-40 questions minimum.** Below 20 is too noisy; above 50 is more work than it's worth at this stage.
2. **Hand-write the questions and ground truths from the source material.** Don't ask the LLM to generate them — circular.
3. **Stratify by difficulty:**
   - ~40% easy (single-chunk lookup, e.g. "what is X")
   - ~40% medium (synthesis, e.g. "compare X and Y")
   - ~20% hard (multi-step, e.g. "given context Z, when would you use X over Y")
4. **Cover all modules** roughly proportionally to syllabus weight.
5. **Include 2-3 "no-answer" questions** — questions whose answer is *not* in the corpus. The system should refuse, not hallucinate. Critical for measuring faithfulness honestly.
6. **Include questions in both English and Hebrew** if you intend to use both.

Add new questions over time but **never delete or modify existing ones** — that breaks comparability with old eval results.

## Eval workflow

### Baseline first
Before any architectural change, capture the current state:

```bash
python scripts/evaluate.py \
    --config config/config.yaml \
    --output data/eval/results/baseline_$(date +%Y%m%d).json
```

### Make the change, then re-run
Use a *different* output file, same test set:

```bash
python scripts/evaluate.py \
    --config config/config_with_reranker.yaml \
    --output data/eval/results/with_reranker.json
```

### Compare
A small script that diffs two result files and prints per-metric deltas. Format:

```
Metric              Baseline    New        Δ        Verdict
faithfulness        0.78        0.91       +0.13    ✓ improved
answer_relevancy    0.82        0.81       -0.01    ≈ neutral
context_precision   0.65        0.79       +0.14    ✓ improved
context_recall      0.71        0.73       +0.02    ≈ neutral
```

A change with one metric clearly up and others neutral is a win. A change with one up and one down is a tradeoff — needs judgment, not auto-merge.

## Interpreting results

| Pattern | What it usually means | What to fix |
|---|---|---|
| Low faithfulness, high answer relevance | Model is confident but ungrounded | Tighten system prompt; add stricter "only from context" instruction |
| High faithfulness, low answer relevance | Answers correct but off-topic | Improve query understanding; check chunking |
| Low context precision, high recall | Retriever is over-fetching | Increase reranker top-n filter; raise quality threshold |
| Low recall, high precision | Retriever is too narrow | Increase top-k; enable HyDE or multi-query |
| All four low | Fundamental issue | Likely chunking — re-examine chunk size and boundaries |

## Tracking results

Store every run with a config hash so you can correlate changes:

```json
{
  "timestamp": "2026-05-07T14:30:00Z",
  "config_hash": "sha256-...",
  "config_summary": {
    "embedder": "BAAI/bge-m3",
    "reranker_enabled": true,
    "hyde_enabled": false,
    "chunk_size": 800
  },
  "metrics": {
    "faithfulness": 0.91,
    "answer_relevancy": 0.85,
    "context_precision": 0.79,
    "context_recall": 0.73
  },
  "per_question": [...]
}
```

Per-question breakdown is gold — when a metric drops, you can drill into *which* questions regressed and inspect them manually.

## Cost discipline

Local LLM eval is "free" in dollars but costly in time. A 30-question eval against GPT-OSS-20B at `medium` reasoning takes ~10-15 minutes. Don't run eval after every micro-change. Run it after meaningful changes:

- New retrieval component
- Embedding model swap
- Chunking strategy change
- Significant prompt revision
- Adding 10+ new files to the corpus

For small tweaks, eyeball-test on 3-5 representative questions first; full eval only if the spot-check looks promising.

## Common pitfalls

- **Test set leakage** — if the LLM-judge is the same model that generated answers, scores are inflated. Mitigation: use deltas across runs, not absolute scores.
- **Optimizing for one metric** — chasing faithfulness while answer_relevance tanks. Look at all four together.
- **Tiny test set** — 5 questions of variance dominate any signal. Minimum 20.
- **Mutable golden answers** — editing the ground truth after seeing the model's answer is data contamination. If a golden answer was wrong, log it, fix it, and start a fresh comparison baseline.
- **Forgetting per-question logs** — aggregate scores hide where the system fails. Always save per-question results.
