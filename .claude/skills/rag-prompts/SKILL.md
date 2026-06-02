---
name: rag-prompts
description: Use this skill when writing or modifying system prompts and prompt templates for the RAG generation layer. Covers task-specific templates for summarization, concept explanation, interview preparation, and topic comparison, plus source attribution and Hebrew/English language handling. Triggers on tasks like "improve the system prompt", "the LLM hallucinates", "make answers more structured", "add a new task type".
---

# RAG Prompts

Prompts are infrastructure. Treat them as you would treat any other config: versioned, testable, comparable across runs.

## Where prompts live

All prompts are in `config/prompts/` as `.txt` files, **not** as Python string literals. Reasons:
- Easy to diff between versions
- Non-developers (even if it's just future-you) can edit them
- Keeps them out of code review noise

Loaded via:

```python
from pathlib import Path

def load_prompt(name: str) -> str:
    return (Path("config/prompts") / f"{name}.txt").read_text()
```

Substitute variables with `str.format(**vars)` — keep templates simple, no Jinja unless you really need conditionals.

## System prompt principles

The system prompt is the contract. It defines:

1. **Role** — what the model is. "You are a study assistant for a graduate-level AI engineering course."
2. **Source grounding** — answer ONLY from the provided context.
3. **Citation requirements** — exact format expected.
4. **Refusal behavior** — what to do when context is insufficient.
5. **Output format** — Markdown, with specific structure.

Bad system prompts are vague ("be helpful and accurate"). Good ones are specific and operational.

## Source grounding (anti-hallucination)

The single most important instruction:

> If the provided context does not contain enough information to answer the question, say so explicitly. Do not use external knowledge. Do not guess.

Without this, models — even reasoning models — will fill gaps from training data. The whole point of RAG is to *not* let it do that.

Followup instruction that helps:

> If you cite a fact, the context must support it. If you find yourself writing a sentence whose source you cannot point to in the context, delete the sentence.

## Citation format

Standardize citations so they're machine-parseable later:

```
[source: <filename>, <section_or_page>]
```

Example: `[source: module2_neural_networks.pdf, §2.2]`

In the prompt:

> Cite every factual claim with `[source: filename, section]`. Use the metadata fields `source` and `section` (or `page` for PDFs).

For multi-source claims, comma-separate: `[source: a.pdf, §1; b.ipynb, cell 4]`.

## Task-specific templates

Each task type has its own template in `config/prompts/<task>.txt`. The system prompt is shared; the user-message template varies.

### `summary.txt` — topic summary

```
Summarize the topic "{topic}" using the context below.

Structure your answer as:
## What it is
2-3 sentences defining the concept.

## Why it matters
2-3 sentences on motivation and applications.

## How it works
A precise but accessible explanation. Use formulas in LaTeX where appropriate.

## When to use it
Concrete scenarios, including any tradeoffs vs alternatives.

## Key terms
Bullet list of 3-5 essential terms with one-line definitions.

Cite every claim. If the context is insufficient on any section, say so for that section instead of guessing.

Context:
{context}
```

### `concept.txt` — concept explanation

```
Explain the concept "{concept}" using the context below.

Format:
- **What it does**: one sentence, plain language.
- **Purpose**: one sentence on what problem it solves.
- **Example**: one short concrete example, ideally with a tiny code snippet or formula.

Be precise. Cite sources.

Context:
{context}
```

### `interview.txt` — interview prep

```
Generate 5 interview-style questions on "{topic}", with answers grounded in the context.

For each question, output a JSON object with:
{{
  "question": "...",
  "answer": "...",  // 2-4 sentences, grounded in context, with citations
  "difficulty": "easy" | "medium" | "hard",
  "follow_up": "..."  // a likely follow-up the interviewer might ask
}}

Output a JSON array of 5 such objects. No other text.

Mix difficulties. Include at least one question that requires connecting two ideas from the context.

Context:
{context}
```

JSON-mode output is valuable here — easier to render in a UI later, easier to evaluate.

### `compare.txt` — comparison

```
Compare {a} and {b} using the context below.

Structure:
## Quick summary
One sentence on each, plus one sentence on the headline difference.

## Comparison table
| Aspect | {a} | {b} |
|---|---|---|
| ... | ... | ... |

Choose 4-6 aspects that are actually relevant to deciding between them, not generic ones.

## When to choose which
2-3 sentences. Be specific about scenarios.

Cite sources throughout.

Context:
{context}
```

## Reasoning models — special considerations

GPT-OSS-20B and similar reasoning models have a `reasoning_effort` parameter (`low` / `medium` / `high`). Calibrate per task:

| Task | Effort |
|---|---|
| Concept explanation | `low` — direct lookup from context |
| Topic summary | `medium` — synthesizing across chunks |
| Interview Q&A generation | `medium` |
| Multi-step comparison | `high` — actual reasoning across sources |

Don't default everything to `high`. It's slow and the extra reasoning is wasted on simple lookups.

The reasoning trace itself is hidden from the user (separate channel in OpenAI-compatible API). Don't try to extract or display it; it's not meant for end-users.

## Hebrew/English handling

The corpus is mostly English; queries may be Hebrew. Default behavior:

> Answer in the same language as the user's question. If the question is in Hebrew, answer in Hebrew. If translation of an English term is uncertain, keep the English term in parentheses next to your translation.

Don't force the model to translate the *context* to Hebrew before answering — quality drops. Let it answer in Hebrew while the context stays English; the model handles cross-lingual answering well.

## Evaluation hook

Every prompt template change should be followed by running the eval suite:

```bash
python scripts/evaluate.py --prompt-version v2 --output data/eval/results/prompts_v2.json
```

Compare with the previous version. If `faithfulness` or `answer_relevance` dropped, the prompt change was net-negative — revert or iterate.

See `rag-evaluation` skill for details.

## Common pitfalls

- **Putting prompt logic in code** — `f"...{role}..."` strings scattered across modules. Centralize in `config/prompts/`.
- **Forgetting to pass metadata to the model** — if the LLM doesn't see `source`, `section`, etc., it can't cite. Format context as: `[source: X, §Y]\n<chunk text>\n\n` for each chunk before injection.
- **Over-prompting** — adding 20 rules in the system prompt. Models follow 5-7 well, the rest get diluted. Prioritize.
- **Inconsistent citation format** — different formats in different templates make later parsing painful. One format, everywhere.
- **No fallback for empty context** — if retrieval returns nothing, the prompt should still produce a graceful "I don't have information on this" rather than hallucinating.
