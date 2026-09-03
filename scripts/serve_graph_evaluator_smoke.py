#!/usr/bin/env python3
"""Serve a deterministic smoke-only GraphRPO evaluator.

This endpoint exercises graph-state scoring and edit-credit plumbing without a
learned evaluator. Its outputs have no semantic meaning and must never be used
for reported training results.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph_rpo import GRAPH_EVALUATOR_SCHEMA_VERSION


def deterministic_probability(question: str, graph_view: str) -> float:
    digest = hashlib.sha256(f"{question}\0{graph_view}".encode("utf-8")).digest()
    unit_value = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return 0.2 + 0.6 * unit_value


def make_app() -> web.Application:
    app = web.Application(client_max_size=16 * 1024**2)

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "schema_version": GRAPH_EVALUATOR_SCHEMA_VERSION,
                "smoke_only": True,
            }
        )

    async def score(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            if payload.get("schema_version") != GRAPH_EVALUATOR_SCHEMA_VERSION:
                raise ValueError("unsupported graph evaluator schema_version")
            items = payload.get("items")
            if not isinstance(items, list) or not items:
                raise ValueError("items must be a non-empty list")
            probabilities = [
                deterministic_probability(
                    str(item.get("question", "")), str(item.get("graph_view", ""))
                )
                for item in items
                if isinstance(item, dict)
            ]
            if len(probabilities) != len(items):
                raise ValueError("every item must be an object")
            return web.json_response(
                {
                    "schema_version": GRAPH_EVALUATOR_SCHEMA_VERSION,
                    "probabilities": probabilities,
                    "smoke_only": True,
                }
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    app.router.add_get("/health", health)
    app.router.add_post("/score", score)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19001)
    args = parser.parse_args()
    web.run_app(make_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
