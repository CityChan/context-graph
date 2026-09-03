#!/usr/bin/env python3
"""Serve a frozen calibrated sequence classifier for GraphRPO graph views."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import torch
from aiohttp import web
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agents.graph_rpo import GRAPH_EVALUATOR_SCHEMA_VERSION, format_graph_evaluator_input


class FrozenGraphEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=args.local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=args.local_files_only
        )
        self.device = torch.device(args.device)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.max_items = args.max_items
        calibration_path = Path(args.model) / "graph_rpo_calibration.json"
        calibration: dict[str, Any] = {}
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.temperature = float(calibration.get("temperature", args.temperature))
        self.positive_label_id = int(calibration.get("positive_label_id", args.positive_label_id))
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    def score(self, items: list[dict[str, Any]]) -> list[float]:
        if not 0 < len(items) <= self.max_items:
            raise ValueError(f"request must contain between 1 and {self.max_items} items")
        texts = [
            format_graph_evaluator_input(
                str(item.get("question", "")), str(item.get("graph_view", ""))
            )
            for item in items
        ]
        probabilities: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits.float() / self.temperature
                if logits.shape[-1] == 1:
                    batch_probabilities = torch.sigmoid(logits[:, 0])
                else:
                    batch_probabilities = torch.softmax(logits, dim=-1)[:, self.positive_label_id]
                probabilities.extend(batch_probabilities.cpu().tolist())
        return probabilities


def make_app(evaluator: FrozenGraphEvaluator) -> web.Application:
    app = web.Application(client_max_size=16 * 1024**2)
    score_lock = asyncio.Lock()

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "schema_version": GRAPH_EVALUATOR_SCHEMA_VERSION})

    async def score(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            if payload.get("schema_version") != GRAPH_EVALUATOR_SCHEMA_VERSION:
                raise ValueError("unsupported graph evaluator schema_version")
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError("items must be a list")
            async with score_lock:
                probabilities = await asyncio.to_thread(evaluator.score, items)
            return web.json_response(
                {
                    "schema_version": GRAPH_EVALUATOR_SCHEMA_VERSION,
                    "probabilities": probabilities,
                }
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    app.router.add_get("/health", health)
    app.router.add_post("/score", score)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--positive-label-id", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    web.run_app(make_app(FrozenGraphEvaluator(args)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
