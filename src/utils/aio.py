from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Iterator, TypeVar

T = TypeVar("T")


async def iterate_in_thread(make_iter: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Consume a blocking iterator in a worker thread without blocking the event loop.

    The LLM stream is a synchronous generator doing network I/O; iterating it
    directly inside an async UI handler would freeze the UI until the answer is
    complete. Items are handed over through a queue as they arrive, and an
    exception raised by the iterator is re-raised here.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def produce() -> None:
        try:
            for item in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    future = loop.run_in_executor(None, produce)
    while (item := await queue.get()) is not done:
        yield item
    await future
