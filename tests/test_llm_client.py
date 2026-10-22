from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation import pipeline as gen_pipeline
from src.generation.llm_client import LLMClient
from src.generation.thinking import ThinkStreamFilter, strip_think
from src.retrieval import transforms


class _FakeCompletions:
    def __init__(self, content="answer", stream_deltas=None):
        self.calls: list[dict] = []
        self.content = content
        self.stream_deltas = stream_deltas

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
                    for d in self.stream_deltas]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def _client(cfg, **fake_kwargs) -> tuple[LLMClient, _FakeCompletions]:
    client = LLMClient()
    fake = _FakeCompletions(**fake_kwargs)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


# --- strip_think / stream filter ---------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("plain answer", "plain answer"),
    ("<think>reasoning</think>\n\nanswer", "answer"),
    ("reasoning from template\n</think>\nanswer", "answer"),  # opening tag was in the prompt
    ("answer <think>cut off by max_tokens", "answer"),
    ("<think>a</think>one <think>b</think>two", "one two"),
])
def test_strip_think(raw, expected):
    assert strip_think(raw) == expected


def test_strip_think_leaves_untagged_text_untouched():
    assert strip_think("  spaced  ") == "  spaced  "


def test_stream_filter_handles_tags_split_across_deltas():
    f = ThinkStreamFilter()
    deltas = ["<thi", "nk>secret ", "reason</th", "ink>\n\nHel", "lo <", "b>"]
    out = "".join(f.feed(d) for d in deltas) + f.flush()
    assert out == "Hello <b>"


def test_stream_filter_passes_plain_text_through():
    f = ThinkStreamFilter()
    assert "".join(f.feed(d) for d in ["a", "b", "c"]) + f.flush() == "abc"


# --- request building --------------------------------------------------------

def test_defaults_send_exactly_what_gpt_oss_got_before(cfg):
    client, fake = _client(cfg)
    client.complete("sys", "q")
    client.helper("sys", "q", max_tokens=128)
    main, helper = fake.calls
    assert main["extra_body"] == {"reasoning_effort": "medium"}
    assert main["temperature"] == 0.1 and main["max_tokens"] == 2048
    assert "top_p" not in main and "presence_penalty" not in main
    assert helper["extra_body"] == {"reasoning_effort": "low"}
    assert helper["temperature"] == 0.0 and helper["max_tokens"] == 128


def test_chat_template_style_switches_thinking_per_call(cfg):
    cfg.generation.thinking.style = "chat_template_kwargs"
    client, fake = _client(cfg)
    client.complete("sys", "q")
    client.helper("sys", "q", max_tokens=64)
    assert fake.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert fake.calls[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_soft_switch_style_appends_switch_to_last_user_message(cfg):
    cfg.generation.thinking.style = "soft_switch"
    client, fake = _client(cfg)
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "a"}]
    client.complete("sys", "q", history=history)
    client.helper("sys", "q", max_tokens=64)
    assert fake.calls[0]["messages"][-1]["content"] == "q /think"
    assert fake.calls[0]["messages"][1]["content"] == "earlier"
    assert history[0]["content"] == "earlier"  # caller's history is not mutated
    assert fake.calls[1]["messages"][-1]["content"] == "q /no_think"
    assert "extra_body" not in fake.calls[1]


def test_sampling_settings_are_sent_when_configured(cfg):
    g = cfg.generation
    g.top_p, g.top_k, g.min_p, g.presence_penalty = 0.95, 20, 0.0, 1.5
    g.thinking.style = "none"
    client, fake = _client(cfg)
    client.complete("sys", "q")
    call = fake.calls[0]
    assert call["top_p"] == 0.95 and call["presence_penalty"] == 1.5
    assert call["extra_body"] == {"top_k": 20, "min_p": 0.0}


def test_answers_are_stripped_of_leaked_reasoning(cfg):
    client, _ = _client(cfg, content="<think>hmm</think>The answer")
    assert client.complete("sys", "q") == "The answer"
    cfg.generation.thinking.strip_tags = False
    client, _ = _client(cfg, content="<think>hmm</think>The answer")
    assert client.complete("sys", "q").startswith("<think>")


def test_stream_strips_reasoning(cfg):
    client, _ = _client(cfg, stream_deltas=["<think>x", "</think>", "Hi", " there"])
    assert "".join(client.stream("sys", "q")) == "Hi there"


# --- callers -----------------------------------------------------------------

def test_answer_uses_per_task_max_tokens(cfg, monkeypatch):
    cfg.generation.task_max_tokens = {"summary": 6000}
    client, fake = _client(cfg)
    monkeypatch.setattr(gen_pipeline, "_client", client)
    gen_pipeline.answer("q", [], task="summary")
    gen_pipeline.answer("q", [], task="concept")
    assert [c["max_tokens"] for c in fake.calls] == [6000, 2048]


def test_helpers_use_configured_budgets(cfg, monkeypatch):
    cfg.generation.helper_max_tokens.condense = 333
    cfg.generation.helper_max_tokens.hyde = 444
    client, fake = _client(cfg, content="rewritten")
    monkeypatch.setattr(transforms, "_client", client)
    assert transforms.condense_query("and that?", [{"role": "user", "content": "x"}]) == "rewritten"
    assert transforms.hyde_transform("q") == "rewritten"
    assert [c["max_tokens"] for c in fake.calls] == [333, 444]


def test_hyde_falls_back_to_query_when_reasoning_eats_the_budget(cfg, monkeypatch):
    client, _ = _client(cfg, content="<think>still thinking when max_tokens ran out")
    monkeypatch.setattr(transforms, "_client", client)
    assert transforms.hyde_transform("what is attention") == "what is attention"
