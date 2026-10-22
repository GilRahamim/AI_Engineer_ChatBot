from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel

# Absolute project root — works regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PathsConfig(BaseModel):
    docs: Path = Path("data/docs")
    chroma_db: Path = Path("data/chroma_db")
    bm25_index: Path = Path("data/bm25_index")
    manifest: Path = Path("data/manifest.json")
    prompts: Path = Path("config/prompts")
    eval_results: Path = Path("data/eval/results")
    cards: Path = Path("summaries/cards")


class IngestionConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_size: int = 100
    supported_extensions: list[str] = [".pdf", ".ipynb"]
    notebook_output_max_lines: int = 20  # longer cell outputs keep head + tail only
    notebook_output_max_chars: int = 2000
    warn_chunks_per_file: int = 500  # log a warning when one file yields more chunks


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    batch_size: int = 32
    device: str = "cuda"  # "cpu" frees VRAM for a larger LLM; fp16 is used on CUDA only
    normalize: bool = True
    max_length: int = 512  # tokens per text


class FusionConfig(BaseModel):
    k: int = 60
    top_n: int = 20


class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5
    device: str = "cuda"


class HyDEConfig(BaseModel):
    enabled: bool = False


class RetrievalConfig(BaseModel):
    dense_top_k: int = 20
    sparse_top_k: int = 20
    collection_name: str = "course_docs"
    fusion: FusionConfig = FusionConfig()
    reranker: RerankerConfig = RerankerConfig()
    hyde: HyDEConfig = HyDEConfig()


class ThinkingConfig(BaseModel):
    # How thinking is switched: "reasoning_effort" (GPT-OSS), "chat_template_kwargs"
    # (enable_thinking flag), "soft_switch" (/think | /no_think in the prompt), "none".
    style: Literal["reasoning_effort", "chat_template_kwargs", "soft_switch", "none"] = "reasoning_effort"
    helper_thinking: bool = False  # think in short helper calls (HyDE, condensing, translation)
    helper_effort: str = "low"  # reasoning_effort for helper calls in "reasoning_effort" style
    soft_switch_on: str = "/think"
    soft_switch_off: str = "/no_think"
    strip_tags: bool = True  # remove <think>...</think> that leaks into answer text


class HelperMaxTokens(BaseModel):
    hyde: int = 256
    condense: int = 128
    translate: int = 128


class GenerationConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    model: str = "openai/gpt-oss-20b"
    api_key: str = "lm-studio"
    temperature: float = 0.1
    # Sampling — None leaves the server/model default. Use the model card's values.
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: int = 2048  # includes reasoning tokens on reasoning models
    task_max_tokens: dict[str, int] = {}  # per-task override, e.g. {"summary": 6000}
    reasoning_effort: str = "medium"
    timeout: float = 1200.0  # seconds per request; long cards on a local model need headroom
    max_retries: int = 0  # SDK retries restart generation from scratch, so keep them off
    thinking: ThinkingConfig = ThinkingConfig()
    helper_temperature: float = 0.0
    helper_max_tokens: HelperMaxTokens = HelperMaxTokens()


class InterviewConfig(BaseModel):
    selection: str = "adaptive"  # "adaptive" | "random"
    # Spaced-repetition interval in days, indexed by score-1 (score 1..5).
    # A question is "due" once this many days have passed since it was last seen.
    interval_days: list[int] = [1, 1, 3, 7, 21]
    unseen_weight: float = 8.0  # selection weight for never-seen questions
    recency_half_life_days: float = 7.0  # how fast a stale question's weight grows
    default_n: int = 5
    review_threshold: int = 3  # scores <= this are eligible for flashcard review
    history_turns: int = 3  # conversational turns kept as chat context


class ChatConfig(BaseModel):
    history_turns: int = 3  # prior (user, assistant) turns passed to the LLM
    condense_followups: bool = True  # rewrite follow-ups into standalone queries


class EvaluationConfig(BaseModel):
    golden_qa: Path = Path("data/eval/golden_qa.jsonl")
    # RAGAS judge. None = the generation model — then switching models also switches
    # the judge and RAGAS scores stop being comparable with older runs.
    judge_model: Optional[str] = None
    judge_base_url: Optional[str] = None
    retrieval_ks: list[int] = [1, 3, 5, 8, 20]  # cutoffs for hit@k / recall@k / precision@k
    label_pool: int = 50  # candidates per retriever shown by scripts/label_sources.py
    label_max_files: int = 15  # candidate files listed per question when labelling


class CardSectionQuery(BaseModel):
    name: str  # shown to the LLM as "retrieved for: <name>"
    query: str  # "{concept}" is replaced with the concept
    file_type: Optional[str] = None  # restrict to one source type, e.g. "notebook"


class CardConfig(BaseModel):
    language: str = "hebrew"  # "hebrew" | "english" — prose language of the card
    sections: list[CardSectionQuery] = [
        CardSectionQuery(name="definition", query="what is {concept}"),
        CardSectionQuery(name="mechanism", query="how does {concept} work step by step"),
        CardSectionQuery(name="formula", query="{concept} formula equation"),
        CardSectionQuery(name="code", query="{concept} python code example", file_type="notebook"),
        CardSectionQuery(name="comparison", query="{concept} compared to alternatives difference"),
        CardSectionQuery(name="tradeoffs", query="{concept} advantages disadvantages limitations"),
        CardSectionQuery(name="applications", query="{concept} applications use cases"),
    ]
    chunks_per_section: int = 4
    max_chunks: int = 24
    use_hyde: bool = False  # templated section queries are already keyword-rich
    translate_concept: bool = True  # Hebrew concept name -> English term for retrieval
    max_tokens: int = 8192
    structured_output: bool = True  # send a JSON schema as response_format
    max_retries: int = 1  # re-ask once when the JSON fails validation
    docx: bool = True
    docx_font: str = "Arial"
    reference_docx: Optional[Path] = None  # Word file whose styles override docx_font


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    ingestion: IngestionConfig = IngestionConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    interview: InterviewConfig = InterviewConfig()
    chat: ChatConfig = ChatConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    card: CardConfig = CardConfig()
    logging: LoggingConfig = LoggingConfig()


_config: Optional[Config] = None


def _resolve_paths(cfg: Config) -> Config:
    """Resolve all PathsConfig fields to absolute paths using PROJECT_ROOT."""
    p = cfg.paths
    cfg.paths = PathsConfig(
        docs=PROJECT_ROOT / p.docs,
        chroma_db=PROJECT_ROOT / p.chroma_db,
        bm25_index=PROJECT_ROOT / p.bm25_index,
        manifest=PROJECT_ROOT / p.manifest,
        prompts=PROJECT_ROOT / p.prompts,
        eval_results=PROJECT_ROOT / p.eval_results,
        cards=PROJECT_ROOT / p.cards,
    )
    cfg.evaluation.golden_qa = PROJECT_ROOT / cfg.evaluation.golden_qa
    if cfg.card.reference_docx is not None:
        cfg.card.reference_docx = PROJECT_ROOT / cfg.card.reference_docx
    return cfg


def load_config(path: str = "config/config.yaml") -> Config:
    global _config
    if _config is not None:
        return _config
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _config = _resolve_paths(Config(**data))
    else:
        _config = _resolve_paths(Config())
    return _config


def get_config() -> Config:
    return load_config()
