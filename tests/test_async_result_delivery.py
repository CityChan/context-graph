import asyncio
import ast
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from envs.async_results import deliver_result


def test_worker_thread_wakes_event_loop_in_debug_mode():
    async def check():
        future = asyncio.get_running_loop().create_future()
        thread = threading.Thread(target=deliver_result, args=(future, {"value": 42}))
        thread.start()
        assert await asyncio.wait_for(future, 2) == {"value": 42}
        thread.join()

    asyncio.run(check(), debug=True)


def test_cancellation_between_scheduling_and_delivery_is_harmless():
    async def check():
        loop = asyncio.get_running_loop()
        errors = []
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        future = loop.create_future()
        deliver_result(future, 1)
        future.cancel()
        await asyncio.sleep(0)
        assert not errors
        later = loop.create_future()
        deliver_result(later, 2)
        assert await later == 2

    asyncio.run(check(), debug=True)


def test_cancelled_server_request_is_removed_from_pending_table():
    # Exercise the real method without loading embedding models or spawning GPU workers.
    source = Path(__file__).resolve().parents[1] / "envs/search_server.py"
    cls = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(node, ast.ClassDef) and node.name == "HighThroughputSearchServer")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "search")
    namespace = {"Dict": dict, "time": time, "SearchRequest": SimpleNamespace,
                 "Full": type("Full", (Exception,), {})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    async def check():
        queued = asyncio.Event()
        server = SimpleNamespace(pending_requests={}, request_queue=SimpleNamespace(put_nowait=lambda _: queued.set()))
        task = asyncio.create_task(namespace["search"](server, "query"))
        await queued.wait()
        assert len(server.pending_requests) == 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert server.pending_requests == {}

    asyncio.run(check(), debug=True)
