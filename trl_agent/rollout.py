"""Async batching adapter from project agent loops to colocated vLLM.

The project agents issue one asynchronous generation request per active turn.
This adapter coalesces all requests that become ready in the same scheduling
wave and executes one serialized ``LLM.generate`` call.  Agent episodes can
therefore remain independent coroutines without paying one vLLM call per
trajectory and per turn.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RolloutResult:
    token_ids: list[int]
    log_probs: list[float]


@dataclass(slots=True)
class _PendingRequest:
    prompt_ids: list[int]
    sampling_params: dict[str, Any]
    future: asyncio.Future[RolloutResult]


def chosen_token_logprobs(token_ids: list[int], logprobs: Any) -> list[float]:
    """Normalize vLLM's chosen-token log-prob representation.

    vLLM versions expose each position either as a mapping from token id to a
    ``Logprob`` object, a scalar-like object, or ``None``.  Missing chosen
    entries are an error: silently substituting zero would corrupt PPO ratios.
    """
    if logprobs is None:
        raise RuntimeError("vLLM did not return requested token log-probabilities")
    if len(logprobs) != len(token_ids):
        raise RuntimeError(
            f"vLLM log-prob/token length mismatch: {len(logprobs)} != {len(token_ids)}"
        )

    result: list[float] = []
    for token_id, position in zip(token_ids, logprobs, strict=True):
        value = position
        if isinstance(position, dict):
            value = position.get(token_id)
            if value is None:
                value = position.get(str(token_id))
        if value is None:
            raise RuntimeError(f"chosen token {token_id} is absent from vLLM logprobs")
        value = getattr(value, "logprob", value)
        result.append(float(value))
    return result


class ColocatedVLLMBatcher:
    """Server-manager-compatible adapter used by :class:`agents.utils.CallLLM`."""

    def __init__(self, llm: Any, *, batch_wait_ms: float = 2.0):
        self.llm = llm
        self.batch_wait_s = max(float(batch_wait_ms), 0.0) / 1000.0
        self._pending: list[_PendingRequest] = []
        self._pending_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()
        self._drain_task: asyncio.Task[None] | None = None
        self.generate_calls = 0
        self.generated_tokens = 0

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
    ) -> RolloutResult:
        del request_id
        if image_data is not None:
            raise NotImplementedError("TRL agent rollout currently supports text-only inputs")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RolloutResult] = loop.create_future()
        request = _PendingRequest(list(prompt_ids), dict(sampling_params), future)
        async with self._pending_lock:
            self._pending.append(request)
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        if self.batch_wait_s:
            await asyncio.sleep(self.batch_wait_s)
        async with self._generation_lock:
            async with self._pending_lock:
                requests, self._pending = self._pending, []
            if not requests:
                return
            try:
                results = await asyncio.to_thread(self._run_batch, requests)
            except BaseException as exc:
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(exc)
            else:
                for request, result in zip(requests, results, strict=True):
                    if not request.future.done():
                        request.future.set_result(result)
        async with self._pending_lock:
            if self._pending:
                self._drain_task = asyncio.create_task(self._drain())

    def _run_batch(self, requests: list[_PendingRequest]) -> list[RolloutResult]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import GuidedDecodingParams

        from agents.structured_outputs import build_vllm_guided_decoding

        prompts = [{"prompt_token_ids": request.prompt_ids} for request in requests]
        params = []
        for request in requests:
            values = dict(request.sampling_params)
            structured_outputs = values.pop("structured_outputs", None)
            if structured_outputs is not None:
                if values.get("guided_decoding") is not None:
                    raise ValueError(
                        "structured_outputs and guided_decoding cannot both be set"
                    )
                values["guided_decoding"] = build_vllm_guided_decoding(
                    structured_outputs, GuidedDecodingParams
                )
            values["logprobs"] = max(int(values.get("logprobs") or 0), 1)
            params.append(SamplingParams(**values))

        lora_requests = None
        lora_ids = list(self.llm.llm_engine.list_loras())
        if lora_ids:
            lora_id = lora_ids[0]
            lora_requests = [
                LoRARequest(
                    lora_name=str(lora_id),
                    lora_int_id=lora_id,
                    lora_path="contextgraph_trl_lora",
                )
                for _ in prompts
            ]

        outputs = self.llm.generate(
            prompts,
            sampling_params=params,
            lora_request=lora_requests,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(requests)} requests"
            )

        results = []
        for request_output in outputs:
            if len(request_output.outputs) != 1:
                raise RuntimeError("agent turn generation requires exactly one sample per request")
            output = request_output.outputs[0]
            token_ids = list(output.token_ids)
            results.append(
                RolloutResult(
                    token_ids=token_ids,
                    log_probs=chosen_token_logprobs(token_ids, output.logprobs),
                )
            )
        self.generate_calls += 1
        self.generated_tokens += sum(len(result.token_ids) for result in results)
        return results
