"""Exercise the production batcher with bounded queues, without GPU imports."""
import ast
import multiprocessing
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Thread
import time
from types import SimpleNamespace

import pytest


def batcher():
    path = Path(__file__).resolve().parents[1] / 'envs/search_server.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'high_speed_batcher')
    namespace = dict(mp=multiprocessing, time=time, Empty=Empty, Full=Full, SearchBatch=SimpleNamespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['high_speed_batcher']


@pytest.mark.parametrize('flush', ['size', 'timeout', 'shutdown'])
def test_full_batch_queue_preserves_requests_and_batcher_survives(flush):
    entered = Event()
    class OutputQueue(Queue):
        def put(self, item, *args, **kwargs):
            if getattr(item, 'requests', None):
                entered.set()
            return super().put(item, *args, **kwargs)

    incoming = Queue()
    outgoing = OutputQueue(maxsize=1)
    outgoing.put('occupied')
    incoming.put('first')
    if flush == 'shutdown':
        incoming.put(None)
    errors = []
    def run():
        try:
            batcher()(incoming, outgoing, max_batch_size=1 if flush == 'size' else 8, batch_timeout=0.01)
        except BaseException as exc:
            errors.append(exc)
    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        assert outgoing.get(timeout=2) == 'occupied'
        assert outgoing.get(timeout=2).requests == ['first']
        if flush != 'shutdown':
            incoming.put('second')
            incoming.put(None)
            assert outgoing.get(timeout=2).requests == ['second']
        thread.join(2)
        assert not thread.is_alive()
        assert not errors
    finally:
        incoming.put(None)
        deadline = time.monotonic() + 2
        while thread.is_alive() and time.monotonic() < deadline:
            try:
                outgoing.get(timeout=0.1)
            except Empty:
                pass
            thread.join(0.1)


def test_backpressure_keeps_batches_bounded_and_delivers_every_request_once():
    blocked = Event()
    class OutputQueue(Queue):
        def put(self, item, *args, **kwargs):
            if getattr(item, 'requests', None):
                blocked.set()
            return super().put(item, *args, **kwargs)
    incoming, outgoing = Queue(), OutputQueue(maxsize=1)
    outgoing.put('occupied')
    for request in range(6):
        incoming.put(request)
    incoming.put(None)
    worker = Thread(target=batcher(), args=(incoming, outgoing),
                    kwargs=dict(max_batch_size=2, batch_timeout=1), daemon=True)
    worker.start()
    try:
        assert blocked.wait(2)
        # The batcher stops reading upstream while the downstream queue is full.
        assert incoming.qsize() == 5
        assert outgoing.get(timeout=2) == 'occupied'
        received = [outgoing.get(timeout=2).requests for _ in range(3)]
        assert received == [[0, 1], [2, 3], [4, 5]]
        worker.join(2)
        assert not worker.is_alive()
    finally:
        incoming.put(None)
        deadline = time.monotonic() + 2
        while worker.is_alive() and time.monotonic() < deadline:
            try:
                outgoing.get(timeout=0.1)
            except Empty:
                pass
            worker.join(0.1)
