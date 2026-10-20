"""Same-checkpoint token-completion transport for Route A self-distillation."""
import copy
import json
from pathlib import Path

from scripts.eval_bcp_qwen38 import completion_budget


class RouteAClient:
    def __init__(self, args, tokenizer, config, audit_path, client=None):
        if client is None:
            import httpx
            endpoint = args.teacher_url.rstrip("/").removesuffix("/v1")
            client = httpx.AsyncClient(base_url=endpoint, timeout=600, trust_env=False)
        self.client, self.args, self.tokenizer, self.config = client, args, tokenizer, config
        self.audit_path = Path(audit_path)

    async def create_completion(self, input_ids, **kwargs):
        from agents.exact_context import token_list
        from agents.structured_outputs import normalize_structured_outputs
        limit = completion_budget(self.config, input_ids, kwargs)
        if limit < 10:
            return None
        sampling = {"temperature": self.args.temperature, "top_p": 0.95}
        sampling.update(kwargs.get("sampling_params") or {})
        body = dict(model=self.args.teacher_model, prompt=list(input_ids), max_tokens=limit,
                    return_token_ids=True, skip_special_tokens=False, seed=self.args.seed,
                    **sampling)
        constraints = kwargs.get("structured_outputs")
        if constraints is not None:
            body["structured_outputs"] = normalize_structured_outputs(constraints)
        response = await self.client.post("/v1/completions", json=body)
        response.raise_for_status()
        data = response.json()
        if data.get("model") != self.args.teacher_model:
            raise ValueError("Route A server returned a different model identity")
        choice = data["choices"][0]
        ids = token_list(choice.get("token_ids"))
        # Keep the original IDs (including thinking and EOS); no re-tokenization.
        content = self.tokenizer.decode(ids, skip_special_tokens=True)
        record = {"context_uid": kwargs.get("uid", "preflight"),
                  "messages": copy.deepcopy(kwargs["messages"]),
                  "input_ids": list(input_ids), "output_ids": ids, "response": content,
                  "max_tokens": limit, "finish_reason": choice.get("finish_reason"),
                  "schema": (constraints or {}).get("json"), "structured_outputs": constraints,
                  "usage": data.get("usage"), "sampling": sampling, "seed": self.args.seed,
                  "requested_model": self.args.teacher_model, "returned_model": data.get("model"),
                  "transport": "vllm_tokens", "native_token_ids": True}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"choices": [{"message": {"content": content, "raw_output_ids": ids,
                 "response_log_probs": [0.0] * len(ids),
                 "extra_data": {"input_ids": list(input_ids)},
                 "metrics": {"usage": data.get("usage", {})}}, "finish_reason": choice.get("finish_reason")}]}

    async def close(self):
        await self.client.aclose()
