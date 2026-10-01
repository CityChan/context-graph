"""Own a rollout environment through initialization, execution and cancellation."""

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager


async def _close(env):
    close = getattr(env, "aclose", None) or getattr(env, "close", None)
    if close is not None:
        result = close()
        if inspect.isawaitable(result):
            await result


@asynccontextmanager
async def managed_environment(env):
    error = None
    try:
        yield env
    except BaseException as exc:
        error = exc
        raise
    finally:
        # A second cancellation must not abandon an in-flight asynchronous close.
        task = asyncio.create_task(_close(env))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        try:
            task.result()
        except BaseException:
            if error is None and not cancelled:
                raise
            logging.getLogger(__name__).exception("Environment cleanup failed during rollout exit")
        if cancelled and error is None:
            raise asyncio.CancelledError
