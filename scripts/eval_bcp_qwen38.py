"""BC-P / GAIA text-only evaluation against local vLLM (no training).

Keep the judge's OpenAI credentials/endpoint separate from the model endpoint.
Send exact agent input IDs and retain generated IDs, including thinking tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation_records import read_evaluation
from scripts.generation_audit import degeneration_stats, has_degenerate_run, require_generation_quality


def select_indices(size, samples, seed):
    if samples == -1:
        return list(range(size))
    if not 1 <= samples <= size:
        raise ValueError("samples must be -1 or between 1 and dataset size")
    return sorted(random.Random(seed).sample(range(size), samples))


def completion_budget(config, input_ids, kwargs):
    maximum = config.prompt_length + config.response_length
    remaining = min(kwargs.get("max_len") or maximum, maximum) - len(input_ids)
    if not kwargs.get("bypass_turn_max_new_tokens", False):
        cap = config.plugin.turn_max_new_tokens
        if cap > 0:
            remaining = min(remaining, cap)
    return min(remaining, kwargs.get("max_new_tokens", remaining))


def validate_dataset(rows, benchmark):
    """Reject an accidental BC-P input or unsupported file tasks for GAIA."""
    if benchmark != "gaia":
        return
    if not rows:
        raise ValueError("Empty GAIA dataset")
    identities = set()
    for row in rows:
        extra = row.get("extra_info") or {}
        if row.get("ability") != "GAIA" or row.get("data_source") != "gaia":
            raise ValueError("GAIA requires the prepared GAIA parquet, not BC-P data")
        if any(str(extra.get(key) or "").strip() for key in ("file_name", "file_path")):
            raise ValueError("GAIA file attachments are unsupported by this text-only evaluator")
        identity = extra.get("instance_id") or extra.get("task_id")
        answer = extra.get("answer")
        if not identity or identity in identities or not isinstance(answer, str) or not answer.strip():
            raise ValueError("GAIA requires unique task IDs and nonempty reference answers")
        identities.add(identity)


class TokenClient:
    def __init__(self, endpoint, model, tokenizer, config, seed, audit_path):
        import httpx
        self.client = httpx.AsyncClient(base_url=endpoint.rstrip("/"), timeout=600, trust_env=False)
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.seed, self.audit_path = seed, audit_path
        self.failed = False

    async def create_completion(self, input_ids, **kwargs):
        from agents.structured_outputs import normalize_structured_outputs
        limit = completion_budget(self.config, input_ids, kwargs)
        if limit < 10:
            return None
        body = dict(model=self.model, prompt=list(input_ids), max_tokens=limit,
                    temperature=0.0, top_p=1.0, seed=self.seed,
                    return_token_ids=True, skip_special_tokens=False)
        if kwargs.get("structured_outputs") is not None:
            body["structured_outputs"] = normalize_structured_outputs(kwargs["structured_outputs"])
        try:
            response = await self.client.post("/v1/completions", json=body)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            ids = choice.get("token_ids")
            if not isinstance(ids, list) or not ids:
                raise RuntimeError("vLLM must return nonempty generated token_ids")
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"input_ids": list(input_ids), "output_ids": ids,
                    "max_tokens": limit, "structured_outputs": body.get("structured_outputs"),
                    "usage": data.get("usage"), "finish_reason": choice.get("finish_reason")}) + "\n")
            return {"choices": [{"message": {"content": text, "raw_output_ids": ids,
                "response_log_probs": [0.0] * len(ids),
                "extra_data": {"input_ids": list(input_ids)},
                "metrics": {"usage": data.get("usage", {})}}}]}
        except Exception:
            self.failed = True
            raise


def config_for(args):
    from omegaconf import OmegaConf
    workflow = {"react": "search", "foldagent": "search_branch",
                "contextgraph": "search_graph"}[args.method]
    # The graph executor reads this even for inference. An empty estimator
    # explicitly leaves training-only GraphRPO credit assignment disabled.
    return OmegaConf.create({"algorithm": {"adv_estimator": ""}, "actor_rollout_ref": {"rollout": {
        "prompt_length": 8192, "response_length": 24576,
        "plugin": {
            "workflow": workflow, "max_turn": 100, "val_max_turn": 100,
            "max_session": 10, "val_max_session": 10, "session_timeout": 3600,
            "branch_len": 24576, "turn_max_new_tokens": 2048,
            "val_response_length": 24576, "process_reward": None,
            "max_traj": 11, "must_finish": False, "double_check": False,
            "must_search": True, "enable_summary": False,
            "final_answer_reserve": 1024, "final_answer_safety_margin": 64,
            "structured_graph_controller": args.method == "contextgraph",
            "controller_owned_tool_formatting": args.method == "contextgraph",
            "controller_action_policy": "structural",
            "contextgraph_memory_mode": args.memory_mode,
            "consolidation_interval": 5, "auto_prune_max_active": 12,
            "structured_memory_enabled": False, "structured_memory_required": False,
            "diagnostic_fix": "none", "capture_model_contexts": True,
            "search_topk_cap": 5, "search_snippet_words": 128,
            "search_snippet_chars": 2000, "open_page_words": 4096,
            "open_page_chars": 48000,
            "apply_chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        }}}})


def tokenizer_preflight(tokenizer, config):
    from agents.utils import AgentContext
    context = AgentContext([{"role": "system", "content": "Use tools."},
                            {"role": "user", "content": "Find evidence."}], tokenizer, config)
    context.context()
    text = "<think>Check the sources carefully.</think>\n<function=search>evidence</function>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    context.append({"role": "assistant", "content": text}, {"choices": [{"message": {
        "raw_output_ids": ids, "response_log_probs": [0.0] * len(ids)}}]})
    evidence = "Branch returned: verified evidence 70468."
    context.append({"role": "user", "content": evidence})
    if evidence not in tokenizer.decode(context.context(), skip_special_tokens=False):
        raise RuntimeError("Tokenizer lost the appended observation")
    context.replace_user_turn(3, "Replacement evidence 70469.")
    if "Replacement evidence 70469." not in tokenizer.decode(context.context(), skip_special_tokens=False):
        raise RuntimeError("Tokenizer lost replacement evidence")


def summarize(root):
    root = Path(root)
    _, results, summary = read_evaluation(root)
    paths = [root / f"requests-{row['source_index']}.jsonl" for row in results]
    summary.update(degeneration_stats(path for path in paths if path.exists()))
    summary["generation_unaudited_records"] = sum(not path.exists() for path in paths)
    if summary["generation_unaudited_records"] and summary["generation_quality_passed"] is True:
        summary["generation_quality_passed"] = None
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    require_generation_quality(summary)
    if summary["execution_errors"] or summary["judge_parse_failures"]:
        raise RuntimeError("Evaluation contains execution/judge failures; do not treat as a clean score")


async def evaluate(args):
    import pandas as pd
    import transformers
    from omegaconf import OmegaConf
    from agents.utils import TaskContext
    from scripts.eval_gaia import _make_dataproto, _metrics_from_output, _process_item_for_workflow

    if not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == "dummy":
        raise RuntimeError("Real judge credentials are required; local model uses a separate endpoint")
    frame = pd.read_parquet(args.data)
    validate_dataset(frame.to_dict("records"), args.benchmark)
    indices = select_indices(len(frame), args.samples, args.seed)
    config = config_for(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer_preflight(tokenizer, config.actor_rollout_ref.rollout)
    root = Path(args.output)
    manifest = {"source_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "benchmark": args.benchmark, "retrieval": "local_bcp_corpus",
        "data_path": str(Path(args.data).resolve()),
        "indices": indices, "rank": args.rank, "method": args.method,
        "model": args.model, "model_path": str(Path(args.model_path).resolve()), "seed": args.seed,
        "judge_model": os.environ.get("JUDGE_MODEL", "gpt-5-nano"),
        "server_execution": {"requested_enforce_eager": os.environ.get("SERVER_ENFORCE_EAGER", "1") == "1"},
        "transformers": transformers.__version__, "config": OmegaConf.to_container(config),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (root / f"manifest-{args.rank}.json").write_text(json.dumps(manifest, indent=2))

    # Keep the same server preflight across all methods.
    probe = TokenClient(args.endpoint, args.model, tokenizer, config.actor_rollout_ref.rollout,
                        args.seed, root / f"preflight-{args.rank}.jsonl")
    try:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Say OK."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        await probe.create_completion(prompt, max_new_tokens=16)
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                  "required": ["ok"], "additionalProperties": False}
        answer = await probe.create_completion(prompt, max_new_tokens=64, structured_outputs={"json": schema})
        assert json.loads(answer["choices"][0]["message"]["content"])["ok"] in (True, False)
    finally:
        await probe.client.aclose()

    semaphore = asyncio.Semaphore(args.workers)
    output_path = root / f"results-{args.rank}.jsonl"
    with output_path.open("x", encoding="utf-8"):
        pass
    async def one(index):
        async with semaphore:
            row = frame.iloc[index].to_dict()
            workflow = config.actor_rollout_ref.rollout.plugin.workflow
            item = _make_dataproto(row, workflow)
            item.meta_info["max_turn"] = 100
            client = TokenClient(args.endpoint, args.model, tokenizer, config.actor_rollout_ref.rollout,
                                 args.seed + index, root / f"requests-{index}.jsonl")
            result = {"source_index": index, "task_id": str(item.non_tensor_batch["uid"][0]),
                      "status": "error", "task_reward": 0.0, "is_finish": False}
            try:
                context = TaskContext(config=config, global_step=0, llm_client=client,
                                      is_train=False, tokenizer=tokenizer)
                output = await _process_item_for_workflow(workflow)(item, context)
                if not output or client.failed:
                    raise RuntimeError("Empty rollout or model request failure")
                reward, shaped, finish, extra = _metrics_from_output(output)
                result.update(status="ok", task_reward=reward, agent_reward=shaped,
                              is_finish=finish, env_stats=extra.get("env_stats", {}))
                (root / f"trajectory-{index}.json").write_text(
                    json.dumps(extra, ensure_ascii=False, default=str), encoding="utf-8")
            except Exception as exc:
                result["error"] = repr(exc)
                traceback.print_exc()
            finally:
                await client.client.aclose()
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, default=str) + "\n")
            print(json.dumps(result, default=str), flush=True)
    await asyncio.gather(*(one(index) for index in indices[args.rank::3]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--benchmark", choices=["bcp", "gaia"], default="bcp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["react", "foldagent", "contextgraph"])
    parser.add_argument("--memory-mode", choices=["legacy", "repaired"], default="repaired")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--model-path")
    parser.add_argument("--endpoint")
    parser.add_argument("--data", default="data/bc_test.parquet")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, choices=range(3), default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.merge:
        summarize(Path(args.output))
    else:
        if not args.method or not args.model_path or not args.endpoint or args.workers < 1:
            parser.error("method, model-path, endpoint and positive workers are required")
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
