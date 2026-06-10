# AI Engineer — Local Learning System

A local RAG (Retrieval-Augmented Generation) system for studying AI/ML course material. Ask questions, get concept explanations with citations, simulate job interviews, and generate study PDFs — all running offline on your own GPU.

**Design philosophy:** slow and accurate beats fast and noisy. Retrieval quality is the top priority.

---

## Features

- **Hybrid retrieval** — dense (ChromaDB + BGE-M3) and sparse (BM25) search fused via Reciprocal Rank Fusion, then reranked by a cross-encoder
- **HyDE query expansion** — generates a hypothetical answer before searching to improve recall on vague questions
- **Follow-up condensing** — rewrites conversational follow-ups into standalone retrieval queries
- **Interview simulator** — adaptive spaced-repetition question selection, RAG-grounded answer evaluation (1–5 stars), session history
- **Study PDF generator** — produces formatted summaries for any topic
- **RAGAS evaluation** — faithfulness, context precision, and context recall metrics to track retrieval quality over time
- **Multilingual** — BGE-M3 and BGE-reranker-v2-m3 support both Hebrew and English

---

## Architecture

```
Query
  └─► Query transform (HyDE / condense follow-up)
        ├─► Dense retrieval   (ChromaDB + BGE-M3)
        └─► Sparse retrieval  (BM25)
              └─► RRF fusion
                    └─► Cross-encoder rerank
                          └─► LLM (GPT-OSS-20B via LM Studio)
                                └─► Answer + inline citations
```

**Ingestion pipeline (offline):**
```
data/docs/  ──►  Loader (PDF / Jupyter)  ──►  Chunker  ──►  BGE-M3
                                                                ├─► ChromaDB
                                                                └─► BM25 index
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| PDF loading | PyMuPDF + pymupdf4llm |
| Notebook loading | nbformat |
| Embeddings | BAAI/bge-m3 (FlagEmbedding) |
| Vector store | ChromaDB |
| Sparse index | bm25s |
| Fusion | Custom RRF (k=60) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | GPT-OSS-20B via LM Studio (OpenAI-compatible API) |
| Evaluation | RAGAS |
| Config | pydantic-settings + YAML |

---

## Hardware requirements

Tested on an NVIDIA RTX 5080 (16 GB VRAM). Approximate VRAM budget:

| Component | VRAM |
|---|---|
| GPT-OSS-20B Q4_K_M (LM Studio) | ~13 GB |
| BGE-M3 embedder | ~1.5 GB |
| BGE-reranker-v2-m3 | ~0.6 GB |
| Reserve | ~1 GB |

The embedder and reranker can run on CPU if VRAM is tight (set `device: cpu` in `config/config.yaml`), at the cost of ingestion and retrieval speed.

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd AI_Engineer_ChatBot

conda create -n myenv python=3.11
conda activate myenv
pip install -e .
```

### 2. Configure LM Studio

- Download and load `openai/gpt-oss-20b` (Q4_K_M quantization)
- Start the local server at `http://localhost:1234/v1`
- Enable Flash Attention, set context length to 16K–32K

The generation layer uses the OpenAI SDK with `base_url="http://localhost:1234/v1"` — no API key needed.

### 3. Add course documents

Place PDF and Jupyter notebook files under `data/docs/` (subdirectories per module are fine):

```
data/docs/
├── Topic 1 - Intro/
├── Topic 2 - NLP/
└── Topic 3 - Deep Learning/
```

### 4. Ingest

```bash
# First-time full ingest
python scripts/ingest.py

# Add new files only (manifest-based deduplication)
python scripts/add_files.py

# Force full rebuild (e.g. after swapping the embedding model)
python scripts/ingest.py --rebuild
```

---

## Usage

### Chat with course material

```bash
python scripts/chat.py
```

Optional flags:

```bash
# Task-specific prompt profiles
python scripts/chat.py --task concept       # explain a concept
python scripts/chat.py --task summary       # summarize a topic
python scripts/chat.py --task interview     # interview-style Q&A

# Pin a topic for better retrieval focus
python scripts/chat.py --task summary --topic "Transformers"
```

The chat keeps a rolling 3-turn history and rewrites follow-up questions into standalone retrieval queries automatically.

### Interview simulator

```bash
python scripts/interview.py

# Filter by topic or difficulty
python scripts/interview.py --topic "NLP" --difficulty hard --n 10

# Random selection instead of adaptive
python scripts/interview.py --mode random
```

Questions are drawn from `data/eval/golden_qa.jsonl`. The adaptive mode uses spaced-repetition intervals — questions you scored low on come back sooner. Scores and gaps are saved to `data/progress.json`.

### Generate a study PDF

```bash
python scripts/generate_pdf.py "Transformers"
# Output: summaries/Transformers.pdf
```

### Review progress

```bash
python scripts/progress.py
```

### Flashcard review

```bash
python scripts/flashcards.py
```

---

## Evaluation

Run RAGAS evaluation before and after any retrieval change to catch regressions:

```bash
# Capture baseline
python scripts/evaluate.py --output data/eval/results/baseline.json

# After your change
python scripts/evaluate.py --output data/eval/results/after_change.json
```

Metrics tracked: faithfulness, context precision, context recall.

The golden Q&A test set lives in `data/eval/golden_qa.jsonl` and is version-controlled.

---

## Configuration

All parameters are in `config/config.yaml` — nothing is hardcoded in source:

```yaml
retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  fusion:
    k: 60          # RRF constant
    top_n: 20
  reranker:
    enabled: true
    top_n: 8
  hyde:
    enabled: true  # hypothetical document expansion

generation:
  base_url: "http://localhost:1234/v1"
  model: "openai/gpt-oss-20b"
  temperature: 0.1
  reasoning_effort: "medium"   # set to "high" for complex interview questions

interview:
  selection: "adaptive"        # "adaptive" | "random"
  interval_days: [1, 1, 3, 7, 21]   # spaced-repetition per score 1–5
```

Prompt templates are plain text files in `config/prompts/` — edit them without touching Python code.

---

## Project structure

```
├── config/
│   ├── config.yaml          # all parameters
│   └── prompts/             # system / task-specific prompt templates
├── data/
│   ├── docs/                # course files (gitignored)
│   ├── chroma_db/           # vector store (gitignored)
│   ├── bm25_index/          # sparse index (gitignored)
│   ├── manifest.json        # ingestion deduplication (committed)
│   └── eval/
│       └── golden_qa.jsonl  # RAGAS test set (committed)
├── src/
│   ├── ingestion/           # loaders, chunkers, manifest, pipeline
│   ├── embedding/           # BGE-M3 wrapper
│   ├── retrieval/           # dense, sparse, fusion, reranker, HyDE, pipeline
│   ├── generation/          # LM Studio client, answer pipeline
│   ├── evaluation/          # RAGAS runner
│   └── utils/               # config loader, logging, math rendering
├── scripts/                 # ingest, chat, interview, evaluate, generate_pdf, ...
└── pyproject.toml
```

---

## Development rules

1. **Never mutate existing chunks** — only add or rebuild. Mutating breaks eval comparability.
2. **No hardcoded parameters** — chunk size, top-k, model names all go through `config/config.yaml`.
3. **Eval before and after every retrieval change** — no metric regression allowed.
4. **Absolute imports only** (`from src.retrieval.dense import ...`).
5. **Type hints required** in all `src/` code.

---

## Implementation phases

| Phase | Focus | Status |
|---|---|---|
| 1. MVP | Ingestion + dense retrieval + LM Studio chat | Complete |
| 2. Hybrid + Rerank | BM25 + RRF + cross-encoder | Complete |
| 3. Quality | HyDE, RAGAS evaluation, smart chunking | Complete |
| 4. Polish | PDF generator, interview simulator, progress tracker | Complete |
