from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from src.evaluation.retrieval_eval import (
    SourceMatcher,
    config_hash,
    config_summary,
    judge_model,
    load_golden_qa,
    score_question,
    summarize,
)
from src.generation.pipeline import answer
from src.ingestion.manifest import load_manifest
from src.retrieval.pipeline import retrieve_stages
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def _build_ragas_llm() -> LangchainLLMWrapper:
    cfg = get_config()
    if cfg.evaluation.judge_model is None:
        logger.warning(
            "RAGAS judge is the generation model (%s); set evaluation.judge_model to keep "
            "scores comparable when the generator changes",
            cfg.generation.model,
        )
    lm = ChatOpenAI(
        base_url=cfg.evaluation.judge_base_url or cfg.generation.base_url,
        api_key=cfg.generation.api_key,
        model=judge_model(),
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
    matcher = SourceMatcher(load_manifest())
    ks = get_config().evaluation.retrieval_ks
    retrieval_scores = []

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

        stages = retrieve_stages(query)
        results = stages["final"]
        scored = score_question(item, stages, matcher, ks)
        retrieval_scores.append(scored)
        # Truncate contexts to keep RAGAS judge prompts within context window
        contexts = [r.text[:400] for r in results]
        generated = answer(query, results)

        pq_entry: dict = {
            "id": qid,
            "question": query,
            "ground_truth": ground_truth,
            "generated_answer": generated,
            "retrieved_sources": [r.source for r in results],
            "expected_sources": item.get("expected_sources", []),
            "retrieval": scored,
            "topic": item.get("topic", ""),
            "difficulty": item.get("difficulty", ""),
            "skip_ragas": skip_ragas,
        }

        if not skip_ragas:
            ragas_indices.append(len(per_question))
            samples.append(SingleTurnSample(
                user_input=query,
                response=generated[:800],
                retrieved_contexts=contexts[:3],
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

    cfg_summary = config_summary()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(cfg_summary),
        "config_summary": cfg_summary,
        "n_questions_total": len(per_question),
        "n_questions_scored": len(ragas_indices),
        "n_questions_labelled": sum(1 for q in questions if q.get("expected_sources")),
        "metrics": {name: float(df[name].mean()) for name in metric_names},
        "retrieval_metrics": summarize(retrieval_scores),
        "per_question": per_question,
    }
