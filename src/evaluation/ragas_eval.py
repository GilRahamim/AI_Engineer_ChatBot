from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.generation.pipeline import answer
from src.retrieval.pipeline import retrieve
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def load_golden_qa(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_ragas_llm() -> LangchainLLMWrapper:
    cfg = get_config()
    lm = ChatOpenAI(
        base_url=cfg.generation.base_url,
        api_key=cfg.generation.api_key,
        model=cfg.generation.model,
        temperature=0.0,
    )
    return LangchainLLMWrapper(lm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    cfg = get_config()
    hf = HuggingFaceEmbeddings(
        model_name=cfg.embedding.model,
        model_kwargs={"device": cfg.embedding.device},
        encode_kwargs={"normalize_embeddings": cfg.embedding.normalize},
    )
    return LangchainEmbeddingsWrapper(hf)


def _config_summary() -> dict:
    cfg = get_config()
    return {
        "embedder": cfg.embedding.model,
        "reranker_enabled": cfg.retrieval.reranker.enabled,
        "reranker_model": cfg.retrieval.reranker.model,
        "dense_top_k": cfg.retrieval.dense_top_k,
        "sparse_top_k": cfg.retrieval.sparse_top_k,
        "fusion_k": cfg.retrieval.fusion.k,
        "reranker_top_n": cfg.retrieval.reranker.top_n,
        "chunk_size": cfg.ingestion.chunk_size,
        "chunk_overlap": cfg.ingestion.chunk_overlap,
    }


def _config_hash(summary: dict) -> str:
    s = json.dumps(summary, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def run_eval(
    golden_path: Path,
    question_ids: list[str] | None = None,
) -> dict[str, Any]:
    questions = load_golden_qa(golden_path)
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]

    logger.info("Running eval on %d questions", len(questions))

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    samples = []
    per_question = []
    # Track which indices in per_question map to RAGAS samples
    ragas_indices: list[int] = []

    for item in questions:
        qid = item["id"]
        query = item["question"]
        ground_truth = item["ground_truth"]
        skip_ragas = item.get("skip_ragas", False)
        logger.info("Processing %s: %s", qid, query[:60])

        results = retrieve(query)
        # Truncate contexts to keep RAGAS judge prompts within context window
        contexts = [r.text[:500] for r in results]
        generated = answer(query, results)

        pq_entry: dict = {
            "id": qid,
            "question": query,
            "ground_truth": ground_truth,
            "generated_answer": generated,
            "retrieved_sources": [r.source for r in results],
            "topic": item.get("topic", ""),
            "difficulty": item.get("difficulty", ""),
            "skip_ragas": skip_ragas,
        }

        if not skip_ragas:
            ragas_indices.append(len(per_question))
            samples.append(SingleTurnSample(
                user_input=query,
                response=generated,
                retrieved_contexts=contexts,
                reference=ground_truth,
            ))

        per_question.append(pq_entry)

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    df = result.to_pandas()

    for df_i, pq_i in enumerate(ragas_indices):
        per_question[pq_i].update(df[metric_names].iloc[df_i].to_dict())

    cfg_summary = _config_summary()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": _config_hash(cfg_summary),
        "config_summary": cfg_summary,
        "n_questions_total": len(per_question),
        "n_questions_scored": len(ragas_indices),
        "metrics": {name: float(df[name].mean()) for name in metric_names},
        "per_question": per_question,
    }
