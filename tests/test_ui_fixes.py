from __future__ import annotations

import ast
import asyncio
import threading
import time
from pathlib import Path

import pytest

from src.utils.aio import iterate_in_thread

ROOT = Path(__file__).resolve().parent.parent


def _collect(make_iter):
    async def run():
        return [item async for item in iterate_in_thread(make_iter)]
    return asyncio.run(run())


def test_items_arrive_in_order():
    assert _collect(lambda: iter(["a", "b", "c"])) == ["a", "b", "c"]


def test_iterator_runs_off_the_event_loop_thread():
    loop_thread = threading.get_ident()
    seen = []

    def gen():
        seen.append(threading.get_ident())
        yield "x"

    _collect(gen)
    assert seen and seen[0] != loop_thread


def test_event_loop_stays_responsive_while_the_iterator_blocks():
    def slow():
        for token in ("one", "two"):
            time.sleep(0.2)  # a blocking network read
            yield token

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        out = [t async for t in iterate_in_thread(slow)]
        tick_task.cancel()
        return out, ticks

    out, ticks = asyncio.run(run())
    assert out == ["one", "two"]
    assert ticks >= 10  # other coroutines kept running during the 0.4 s of blocking


def test_iterator_errors_reach_the_consumer():
    def broken():
        yield "partial"
        raise ConnectionError("LM Studio went away")

    with pytest.raises(ConnectionError):
        _collect(broken)


def test_chat_cli_offers_every_task_prompt():
    # Read TASK_CHOICES from the source: importing chat.py loads the model stack.
    tree = ast.parse((ROOT / "scripts" / "chat.py").read_text(encoding="utf-8"))
    choices = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TASK_CHOICES"
    )
    assert "glossary" in choices
    prompts = ROOT / "config" / "prompts"
    for task in choices:
        if task != "default":  # default uses system.txt
            assert (prompts / f"{task}.txt").exists(), task
