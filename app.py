"""Chainlit UI for the AI Engineer Course RAG system."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

import chainlit as cl

from src.generation.pipeline import answer
from src.retrieval.pipeline import retrieve
from src.utils.config import load_config
from src.utils.logging import setup_logging

setup_logging()
load_config()

TASK_DESCRIPTIONS = {
    "default": "General Q&A",
    "concept": "Concept Explanation",
    "interview": "Interview Preparation",
    "summary": "Topic Summary",
}


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "**AI Engineer Course Assistant**\n\n"
            "Ask me anything from the course materials.\n\n"
            "**Task profiles** (type `/task <name>` to switch):\n"
            "- `default` — General Q&A *(active)*\n"
            "- `concept` — Deep concept explanation\n"
            "- `interview` — Interview-style answer\n"
            "- `summary` — Topic summary\n"
        )
    ).send()
    cl.user_session.set("task", "default")


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content.strip()

    # Handle /task switch
    if query.startswith("/task "):
        task = query[6:].strip().lower()
        if task in TASK_DESCRIPTIONS:
            cl.user_session.set("task", task)
            await cl.Message(content=f"Switched to **{TASK_DESCRIPTIONS[task]}** mode.").send()
        else:
            await cl.Message(
                content=f"Unknown task `{task}`. Available: {', '.join(TASK_DESCRIPTIONS)}"
            ).send()
        return

    task = cl.user_session.get("task", "default")

    # Retrieve
    async with cl.Step(name="Retrieving course material") as step:
        results = retrieve(query)
        if not results:
            step.output = "No relevant documents found."
            await cl.Message(content="No relevant documents found in the course materials.").send()
            return
        sources = list({Path(r.source).name for r in results})
        step.output = f"{len(results)} chunks from: {', '.join(sources[:4])}"

    # Generate
    async with cl.Step(name=f"Generating answer [{TASK_DESCRIPTIONS[task]}]"):
        response = answer(query, results, task=task)

    # Sources footer
    source_lines = "\n".join(f"- {Path(r.source).name}" for r in results)
    footer = f"\n\n---\n**Sources ({len(results)} chunks):**\n{source_lines}"

    await cl.Message(content=response + footer).send()
