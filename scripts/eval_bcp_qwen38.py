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


def select_indices(size, samples, seed, indices_file=None):
    if indices_file:
        if samples != -1:
            raise ValueError("Explicit indices require samples=-1")
        indices = json.loads(Path(indices_file).read_text(encoding="utf-8"))
        if (not isinstance(indices, list) or not indices
                or any(type(i) is not int or not 0 <= i < size for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Indices must be a nonempty list of unique in-range integers")
        return sorted(indices)
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
                "raw_output_text": self.tokenizer.decode(ids, skip_special_tokens=False),
                "response_log_probs": [0.0] * len(ids),
                "extra_data": {"input_ids": list(input_ids)},
                "metrics": {"usage": data.get("usage", {})}}, "finish_reason": choice.get('finish_reason')}]}
        except Exception:
            self.failed = True
            raise


def config_for(args):
    from omegaconf import OmegaConf
    window = int(os.environ.get('EVAL_CONTEXT_LENGTH', '32768'))
    if window not in (32768, 65536):
        raise ValueError('EVAL_CONTEXT_LENGTH must be 32768 or 65536')
    guard_flag = os.environ.get('GRAPH_RETRY_GUARD', '0')
    if guard_flag not in ('0', '1'):
        raise ValueError('GRAPH_RETRY_GUARD must be 0 or 1')
    if guard_flag == '1' and args.method != 'contextgraph':
        raise ValueError('GRAPH_RETRY_GUARD only applies to contextgraph')
    max_turn = int(os.environ.get('EVAL_MAX_TURN', '100'))
    session_timeout = int(os.environ.get('EVAL_SESSION_TIMEOUT', '3600'))
    episode_timeout = int(os.environ.get('EVAL_EPISODE_TIMEOUT', '0'))
    no_tool_limit = int(os.environ.get('EVAL_NO_TOOL_LIMIT', '0'))
    task_recovery = os.environ.get('EVAL_ACTION_RECOVERY', '0') == '1'
    if min(max_turn, session_timeout) < 1 or min(episode_timeout, no_tool_limit) < 0:
        raise ValueError('Invalid evaluation turn/time/no-tool limits')
    if (no_tool_limit or task_recovery) and (args.method != 'contextgraph' or getattr(args, 'graph_action_protocol', 'controller') != 'controller'):
        raise ValueError('EVAL_NO_TOOL_LIMIT currently supports controller ContextGraph only')
    workflow = {"react": "search", "foldagent": "search_branch", "agentfold": "search_agentfold",
                "contextgraph": "search_graph", "supo": "search_supo",
                "memobrain": "search_memobrain", "amem": "search_amem"}[args.method]
    # The graph executor reads this even for inference. An empty estimator
    # explicitly leaves training-only GraphRPO credit assignment disabled.
    config = OmegaConf.create({"algorithm": {"adv_estimator": ""}, "actor_rollout_ref": {"rollout": {
        "prompt_length": 8192, "response_length": window - 8192,
        "plugin": {
            "workflow": workflow, "max_turn": max_turn, "val_max_turn": max_turn,
            "max_session": 10, "val_max_session": 10, "session_timeout": session_timeout,
            "eval_episode_timeout": episode_timeout, "inference_no_tool_limit": no_tool_limit,
            "inference_task_action_recovery": task_recovery,
            "eval_require_task_tools": os.environ.get('EVAL_REQUIRE_TASK_TOOLS', '0') == '1',
            "branch_len": window - 8192, "turn_max_new_tokens": 2048,
            "graph_retry_guard": guard_flag == '1',
            "graph_retry_failure_limit": int(os.environ.get('GRAPH_RETRY_FAILURE_LIMIT', '3')),
            "val_response_length": window - 8192, "process_reward": None,
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
    if args.method == "supo":
        from agents.supo_agent import settings
        config.actor_rollout_ref.rollout.plugin.update(settings(args))
    elif args.method in {"memobrain", "amem"}:
        from agents.graph_memory_baselines import settings
        config.actor_rollout_ref.rollout.plugin.update(settings(args))
    from agents.explicit_graph_tools import configure_evaluation_protocol
    return configure_evaluation_protocol(config, args.method, getattr(args, 'graph_action_protocol', 'controller'))


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
    manifests, results, summary = read_evaluation(root)
    paths = [root / f"requests-{row['source_index']}.jsonl" for row in results]
    summary.update(degeneration_stats(path for path in paths if path.exists()))
    summary["generation_unaudited_records"] = sum(not path.exists() for path in paths)
    if summary["generation_unaudited_records"] and summary["generation_quality_passed"] is True:
        summary["generation_quality_passed"] = None
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    require_generation_quality(summary)
    if summary["format_retry_failures"]:
        raise RuntimeError("Evaluation contains tool-format retry failures; do not treat as a clean score")
    if summary["summary_format_failures"]:
        raise RuntimeError("Evaluation contains summary-format failures; do not treat as a clean score")
    if summary["execution_errors"] or summary["judge_parse_failures"]:
        raise RuntimeError("Evaluation contains execution/judge failures; do not treat as a clean score")
    plugin = manifests[0].get('config', {}).get('actor_rollout_ref', {}).get('rollout', {}).get('plugin', {})
    if plugin.get('eval_require_task_tools', False) and summary['zero_tool_tasks']:
        raise RuntimeError('Smoke contains zero-task-tool episodes; inspect saved responses before running the next arm')


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
    indices = select_indices(len(frame), args.samples, args.seed, getattr(args, "indices_file", None))
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
    if getattr(args, "indices_file", None):
        manifest["selection"] = {"kind": "explicit_source_indices", "path": str(Path(args.indices_file).resolve())}
    if args.method == "agentfold":
        from agents.agentfold_agent import PROVENANCE
        manifest["baseline_protocol"] = PROVENANCE
    elif args.method == "supo":
        from agents.supo_agent import PROVENANCE, settings
        manifest["baseline_protocol"] = {**PROVENANCE, **settings(config.actor_rollout_ref.rollout.plugin)}
    elif args.method in {"memobrain", "amem"}:
        from agents.memory_baseline_agent import provenance
        from agents.graph_memory_baselines import settings
        manifest["baseline_protocol"] = provenance(args.method, settings(config.actor_rollout_ref.rollout.plugin))
    from agents.token_budget import TokenBudget
    budget_protocol = TokenBudget.from_config(config.actor_rollout_ref.rollout).metadata()
    manifest['budget_protocol'] = budget_protocol
    if 'baseline_protocol' in manifest:
        manifest['baseline_protocol'] = {**manifest['baseline_protocol'], **budget_protocol}
        if args.method in ('memobrain', 'amem'):
            manifest['baseline_protocol']['finalizer_enabled'] = bool(config.actor_rollout_ref.rollout.plugin.final_answer_reserve)
            manifest['baseline_protocol']['turns'] = 'all actor and memory requests; final answer slot reserved'
    manifest_path = root / f"manifest-{args.rank}.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        old_mode = previous['config']['actor_rollout_ref']['rollout']['plugin'].get('token_budget_mode', 'legacy_cumulative')
        if old_mode != budget_protocol['token_budget_mode']:
            raise ValueError('Budget protocol changed; use a fresh output directory')
    manifest_path.write_text(json.dumps(manifest, indent=2))

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
        if getattr(args, 'graph_action_protocol', 'controller') == 'explicit_tools':
            from agents.action_contract import ActionContract
            contract = ActionContract('LocalSearch', None)
            schema = contract.schema()
            action_prompt = tokenizer.apply_chat_template([{'role': 'user', 'content':
                'Return {"tool":"graph_read","arguments":{}}.'}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            answer = await probe.create_completion(action_prompt, max_new_tokens=256,
                                                    structured_outputs={'json': schema})
            contract.decode(answer['choices'][0]['message']['content'], schema)
            print('Typed action schema preflight passed')
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
            plugin = config.actor_rollout_ref.rollout.plugin
            item.meta_info["max_turn"] = plugin.val_max_turn
            client = TokenClient(args.endpoint, args.model, tokenizer, config.actor_rollout_ref.rollout,
                                 args.seed + index, root / f"requests-{index}.jsonl")
            result = {"source_index": index, "task_id": str(item.non_tensor_batch["uid"][0]),
                      "status": "error", "task_reward": 0.0, "is_finish": False}
            episode_started = asyncio.get_running_loop().time()
            try:
                context = TaskContext(config=config, global_step=0, llm_client=client,
                                      is_train=False, tokenizer=tokenizer)
                rollout = _process_item_for_workflow(workflow)(item, context)
                output = await asyncio.wait_for(rollout, timeout=plugin.eval_episode_timeout or None)
                if not output or client.failed:
                    raise RuntimeError("Empty rollout or model request failure")
                reward, shaped, finish, extra = _metrics_from_output(output)
                result.update(status="ok", task_reward=reward, agent_reward=shaped,
                              is_finish=finish, env_stats=extra.get("env_stats", {}))
                (root / f"trajectory-{index}.json").write_text(
                    json.dumps(extra, ensure_ascii=False, default=str), encoding="utf-8")
            except asyncio.TimeoutError as exc:
                elapsed = asyncio.get_running_loop().time() - episode_started
                wall_timeout = plugin.eval_episode_timeout > 0 and elapsed >= plugin.eval_episode_timeout
                result['error'] = 'episode_wall_timeout' if wall_timeout else repr(exc)
                result['env_stats'] = {'hit_episode_wall_timeout': int(wall_timeout)}
                print(f'EPISODE_TIMEOUT source_index={index} wall_limit={wall_timeout}; partial requests remain in requests-{index}.jsonl', flush=True)
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
    parser.add_argument("--method", choices=["react", "foldagent", "contextgraph", "agentfold", "supo", "memobrain", "amem"])
    parser.add_argument("--memory-helper-max-tokens", type=int, default=1024)
    parser.add_argument("--memobrain-recall-interval", type=int, default=5)
    parser.add_argument("--memobrain-context-threshold", type=int, default=16384)
    parser.add_argument("--amem-topk", type=int, default=5)
    parser.add_argument("--amem-memory-tokens", type=int, default=4096)
    parser.add_argument("--supo-context-threshold", type=int, default=16384)
    parser.add_argument("--supo-max-summaries", type=int, default=2)
    parser.add_argument("--supo-summary-max-tokens", type=int, default=1024)
    parser.add_argument("--memory-mode", choices=["legacy", "repaired"], default="repaired")
    parser.add_argument('--graph-action-protocol', choices=['controller', 'explicit_tools'], default='controller')
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--model-path")
    parser.add_argument("--endpoint")
    parser.add_argument("--data", default="data/bc_test.parquet")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--indices-file", help="JSON list of original dataset row indices; requires --samples=-1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, choices=range(3), default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.graph_action_protocol == 'explicit_tools' and args.method != 'contextgraph':
        parser.error('explicit_tools requires --method contextgraph')
    if args.merge:
        summarize(Path(args.output))
    else:
        if not args.method or not args.model_path or not args.endpoint or args.workers < 1:
            parser.error("method, model-path, endpoint and positive workers are required")
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
