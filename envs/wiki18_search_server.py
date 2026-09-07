#!/usr/bin/env python3
"""Serve the Search-R1 Wiki-18 E5/FAISS index through ContextGraph's API.

The retrieval math and corpus layout follow Search-R1's Apache-2.0
``retrieval_server.py``.  This implementation is local to this repository and
adds request batching plus the ``/search`` and ``/open`` endpoints expected by
``envs.local_search``; it does not require a Search-R1 or SkillRL checkout.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    k: int = 20


class OpenRequest(BaseModel):
    docid: str | None = None
    url: str | None = None


def split_contents(contents: str) -> tuple[str, str]:
    """Split Search-R1's ``title\npassage`` corpus representation."""
    lines = str(contents or "").splitlines()
    if not lines:
        return "", ""
    return lines[0].strip('"'), "\n".join(lines[1:])


def truncate_words(text: str, limit: int) -> str:
    words = str(text or "").split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + "\n[Document is truncated.]"


class Wiki18Retriever:
    """Exact dense retrieval over the published Search-R1 Wiki-18 index."""

    def __init__(self, index_path: str, corpus_path: str, model_path: str, use_gpu: bool):
        import faiss
        import torch
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer

        self.faiss = faiss
        self.torch = torch
        self.corpus = load_dataset(
            "json", data_files=corpus_path, split="train", num_proc=8
        )
        self.index = faiss.read_index(index_path)
        if self.index.ntotal != len(self.corpus):
            raise RuntimeError(
                f"index/corpus mismatch: index={self.index.ntotal}, corpus={len(self.corpus)}"
            )
        if use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("--faiss-gpu requested but CUDA is unavailable")
            options = faiss.GpuMultipleClonerOptions()
            options.useFloat16 = True
            options.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=options)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model.to(self.device)
        if use_gpu:
            self.model.half()

    def encode(self, queries: list[str]):
        torch = self.torch
        prefixed = [f"query: {query}" for query in queries]
        inputs = self.tokenizer(
            prefixed,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs, return_dict=True)
            mask = inputs["attention_mask"].unsqueeze(-1).bool()
            hidden = output.last_hidden_state.masked_fill(~mask, 0.0)
            embeddings = hidden.sum(dim=1) / mask.sum(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy()

    def search_batch(self, queries: list[str], topk: int) -> list[list[dict[str, Any]]]:
        embeddings = self.encode(queries)
        scores, row_ids = self.index.search(embeddings, topk)
        batches: list[list[dict[str, Any]]] = []
        for hit_ids, hit_scores in zip(row_ids.tolist(), scores.tolist()):
            results = []
            for row_id, score in zip(hit_ids, hit_scores):
                if row_id < 0:
                    continue
                doc = self.corpus[int(row_id)]
                contents = str(doc.get("contents", ""))
                title, passage = split_contents(contents)
                snippet = f"{title}\n{passage}" if title else passage
                results.append(
                    {
                        "docid": str(row_id),
                        "url": f"wiki18://{row_id}",
                        "text": truncate_words(snippet, 1000),
                        "score": float(score),
                    }
                )
            batches.append(results)
        return batches

    def open_document(self, docid: str | None) -> dict[str, Any] | None:
        try:
            row_id = int(docid)
        except (TypeError, ValueError):
            return None
        if row_id < 0 or row_id >= len(self.corpus):
            return None
        doc = self.corpus[row_id]
        contents = str(doc.get("contents", ""))
        return {
            "docid": str(row_id),
            "url": f"wiki18://{row_id}",
            "text": truncate_words(contents, 15000),
        }


@dataclass
class PendingSearch:
    query: str
    topk: int
    future: asyncio.Future


class BatchedSearchEngine:
    def __init__(self, retriever: Wiki18Retriever, max_batch_size: int, timeout_ms: float):
        self.retriever = retriever
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_ms / 1000.0
        self.queue: asyncio.Queue[PendingSearch] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.request_count = 0

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def submit(self, query: str, topk: int) -> list[dict[str, Any]]:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(PendingSearch(query=query, topk=topk, future=future))
        return await asyncio.wait_for(future, timeout=120.0)

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.timeout_seconds
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            try:
                max_topk = max(item.topk for item in batch)
                results = await asyncio.to_thread(
                    self.retriever.search_batch,
                    [item.query for item in batch],
                    max_topk,
                )
                self.request_count += len(batch)
                for item, item_results in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(item_results[: item.topk])
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)


def serve(args: argparse.Namespace) -> None:
    import uvicorn
    from fastapi import FastAPI, HTTPException

    started = time.time()
    retriever = Wiki18Retriever(
        args.index_path, args.corpus_path, args.model_path, args.faiss_gpu
    )
    engine = BatchedSearchEngine(retriever, args.max_batch_size, args.batch_timeout_ms)
    app = FastAPI()

    @app.on_event("startup")
    async def startup_event() -> None:
        await engine.start()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await engine.stop()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "backend": "wiki18_e5_faiss",
            "corpus_size": len(retriever.corpus),
            "index_size": int(retriever.index.ntotal),
        }

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return {
            "backend": "wiki18_e5_faiss",
            "corpus_size": len(retriever.corpus),
            "requests": engine.request_count,
            "pending": engine.queue.qsize(),
            "uptime_seconds": time.time() - started,
        }

    @app.post("/search")
    async def search(request: QueryRequest) -> dict[str, Any]:
        if request.k < 1 or request.k > 100:
            raise HTTPException(status_code=400, detail="k must be between 1 and 100")
        begin = time.time()
        results = await engine.submit(request.query, request.k)
        return {"results": results, "took_ms": (time.time() - begin) * 1000.0}

    @app.post("/open")
    async def open_document(request: OpenRequest) -> dict[str, Any]:
        docid = request.docid
        if docid is None and request.url and request.url.startswith("wiki18://"):
            docid = request.url.removeprefix("wiki18://")
        result = await asyncio.to_thread(retriever.open_document, docid)
        if result is None:
            raise HTTPException(status_code=404, detail="document not found")
        return {"results": [result], "took_ms": 0.0}

    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Search-R1 Wiki-18 retrieval")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18999)
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--batch-timeout-ms", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
