# AI Engineer Course — Local Learning System

A local RAG system for summarizing AI/ML course material, explaining concepts, and preparing for interviews.
Built for high retrieval quality, not production deployment — slow and accurate beats fast and noisy.

---

## Architecture at a glance

Two pipelines:

**Ingestion (offline)** — `docs/ → loader → chunker → embedder → {ChromaDB, BM25 index}`

**Query (online)** — `query → query transform → hybrid retrieval (dense ‖ sparse) → RRF fusion → cross-encoder rerank → LLM → answer + citations`

Always prioritize retrieval quality over generation quality. A good LLM cannot save a bad context.

---

## Tech stack

| Layer | Tool | Note |
|---|---|---|
| Orchestration | LangChain | Familiar, rich ecosystem |
| PDF loading | PyMuPDF + pymupdf4llm | Structural extraction to Markdown |
| Notebook loading | nbformat | Separates cells by type |
| Chunking | RecursiveCharacterTextSplitter + custom | Structure-aware |
| Embeddings | BGE-M3 (via FlagEmbedding) | Multilingual (Hebrew + English), hybrid native |
| Vector store | ChromaDB | Persistence + metadata filters |
| Sparse index | bm25s | Faster than rank_bm25 |
| Fusion | RRF (custom) | $\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$, $k=60$ |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder, supports Hebrew |
| LLM | **GPT-OSS-20B** via LM Studio | Reasoning model, ~13GB VRAM |
| Eval | RAGAS | Faithfulness, context precision/recall |
| Config | pydantic-settings | Type-safe config |

GPU: NVIDIA RTX 5080, 16GB VRAM. Rough budget:
- LLM (GPT-OSS-20B Q4_K_M): ~13GB
- Embedder (BGE-M3): ~1.5GB
- Reranker (BGE-reranker-v2-m3): ~0.6GB
- Reserve: ~1GB

---

## Project structure

```
course_rag/
├── CLAUDE.md                     <- you are here
├── .claude/skills/               <- task-specific skills
│   ├── document-ingestion/SKILL.md
│   ├── hybrid-retrieval/SKILL.md
│   ├── rag-prompts/SKILL.md
│   └── rag-evaluation/SKILL.md
├── config/
│   ├── config.yaml               <- all parameters, no hardcoding allowed
│   └── prompts/                  <- prompt templates (text, not Python)
│       ├── system.txt
│       ├── summary.txt
│       ├── concept.txt
│       └── interview.txt
├── data/
│   ├── docs/                     <- course files (raw, read-only)
│   ├── chroma_db/                <- persistent vector store
│   ├── bm25_index/               <- persistent sparse index
│   ├── manifest.json             <- hash tracking of ingested files
│   └── eval/
│       ├── golden_qa.jsonl       <- test set for RAGAS
│       └── results/              <- eval run results
├── src/
│   ├── ingestion/
│   │   ├── loaders.py            <- PDF, notebook loaders
│   │   ├── chunkers.py           <- structure-aware chunking
│   │   ├── manifest.py           <- deduplication
│   │   └── pipeline.py
│   ├── embedding/
│   │   └── embedder.py           <- BGE-M3 wrapper
│   ├── retrieval/
│   │   ├── dense.py              <- ChromaDB client
│   │   ├── sparse.py             <- BM25 client
│   │   ├── fusion.py             <- RRF
│   │   ├── reranker.py           <- cross-encoder
│   │   ├── transforms.py         <- HyDE, multi-query
│   │   └── pipeline.py           <- end-to-end retrieval
│   ├── generation/
│   │   ├── llm_client.py         <- LM Studio (OpenAI-compatible)
│   │   └── pipeline.py
│   ├── evaluation/
│   │   └── ragas_eval.py
│   └── utils/
│       ├── config.py             <- pydantic-settings loader
│       └── logging.py
├── scripts/
│   ├── ingest.py                 <- bulk ingestion
│   ├── add_files.py              <- incremental ingestion
│   ├── chat.py                   <- CLI interface
│   ├── generate_pdf.py           <- formatted study PDFs
│   └── evaluate.py               <- RAGAS run
├── notebooks/                    <- experiments only, not production code
├── tests/
└── pyproject.toml
```

---

## Local LLM setup (GPT-OSS-20B)

LM Studio config:
- Model: `openai/gpt-oss-20b` (Q4_K_M)
- Context length: 16K-32K (start at 16K, increase as needed)
- Reasoning effort: `medium` (`high` only for complex interview questions)
- Server: `http://localhost:1234/v1` (OpenAI-compatible)
- Flash Attention: enabled

The client in `src/generation/llm_client.py` uses the OpenAI SDK with `base_url="http://localhost:1234/v1"`.

---

## Conventions

### Code
- **Type hints required** in every production function (`src/`). Optional in `scripts/` and `notebooks/`.
- **Pydantic models** for data structures passed between layers (Chunk, RetrievalResult, etc.).
- **Logging instead of print** in `src/`. Scripts may use print.
- **No hardcoded paths** — everything goes through `config/config.yaml`.

### Imports
- Absolute imports only (`from src.retrieval.dense import ...`).
- `pyproject.toml` with `[tool.setuptools.packages.find]` makes `src/` an installable package.

### Git
- `data/chroma_db/`, `data/bm25_index/`, `data/eval/results/` — gitignored.
- `data/docs/` — gitignored (course files do not belong in git).
- `data/manifest.json` and `data/eval/golden_qa.jsonl` — committed.

### Commits
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.
- Small commits. Structural changes (e.g. swapping the embedder) = separate commit.

---

## Common workflows

### Adding new course files
1. Place files in `data/docs/` (subdirectory per module if relevant).
2. Run `python scripts/add_files.py` — it detects only new files via the manifest.
3. If you swapped the embedding model — full rebuild required: `python scripts/ingest.py --rebuild`.

-> Details in `.claude/skills/document-ingestion/SKILL.md`

### Asking a question (chat mode)
```bash
python scripts/chat.py
# or with a specific profile:
python scripts/chat.py --task interview --topic "Transformers"
```

-> Details on prompts in `.claude/skills/rag-prompts/SKILL.md`

### Modifying the retrieval pipeline
Before any change — run eval on the current state to get a baseline:
```bash
python scripts/evaluate.py --output data/eval/results/baseline.json
```
After the change — run again and compare.

-> Details in `.claude/skills/hybrid-retrieval/SKILL.md` and `.claude/skills/rag-evaluation/SKILL.md`

### Generating a study PDF
```bash
python scripts/generate_pdf.py "Transformers"
# creates summaries/Transformers.pdf
```

---

## Hard rules — do not break

1. **Never modify existing chunks in the DB** — only add new ones or do a full rebuild. Silently changing chunks breaks eval comparability.
2. **No hardcoded parameters** — chunk size, top-k, model names — everything goes through config.
3. **Every retrieval pipeline change requires eval before and after**. Goal: no metric regression.
4. **The model is only the LLM, not an AI agent**. Do not let it perform actions (write files, call APIs) without explicit user approval.
5. **Never expose API keys in logs**. Even localhost keys — keep clean habits.

---

## Skills index

| Skill | When to load |
|---|---|
| `document-ingestion` | Adding files, changing loader/chunker, debugging ingestion |
| `hybrid-retrieval` | Implementing/changing retrieval, debugging search results |
| `rag-prompts` | Writing/changing system prompts and templates |
| `rag-evaluation` | Building test set, running RAGAS, analyzing results |

---

## Implementation phases

| Phase | Duration | Output |
|---|---|---|
| **1. MVP** | 3-4 days | Ingestion + dense retrieval + LM Studio. No hybrid yet. |
| **2. Hybrid + Rerank** | 2-3 days | BM25 + RRF + cross-encoder. This is where the big quality jump happens. |
| **3. Quality** | 3-4 days | Smart chunking, HyDE, RAGAS evaluation. We measure. |
| **4. Polish** | 3 days | PDF generator, interview simulator, progress tracker. |

Always build phase 1 end-to-end before adding complexity. Do not start phase 2 before the pipeline works end-to-end.
