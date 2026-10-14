#!/usr/bin/env python3
"""Collect real BC-P or ScienceWorld training episodes with a DeepSeek teacher.

Teacher credentials are separate from OPENAI_* (the existing BC-P grader).
Run against an existing retrieval service; the shell wrapper can start one.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_bcp_qwen38 import completion_budget, config_for

PROFILE = "bcp_explicit_teacher_v1"
SCIENCEWORLD_PROFILE = "scienceworld_explicit_teacher_v1"
AUTO_PROFILE = "contextgraph_auto_teacher_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def question(row):
    extra = row.get("extra_info") or {}
    if row.get("ability") == "ScienceWorld@real":
        name, variation = extra.get("task_name"), extra.get("variation_idx")
        if not isinstance(name, str) or not name or not isinstance(variation, int) or isinstance(variation, bool):
            raise ValueError("ScienceWorld requires task_name and integer variation_idx")
        return f"scienceworld:{name}:{variation}"
    query, answer = extra.get("query"), extra.get("answer")
    if not isinstance(query, str) or not query.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError("BC-P rows require nonempty extra_info.query and answer")
    if row.get("ability") != "LocalSearch":
        raise ValueError("Expected BC-P LocalSearch rows")
    return " ".join(query.split()).casefold()


def select_training_rows(rows, heldout, start, samples, seed, selection="uniform"):
    if selection not in {"uniform", "task_balanced"}:
        raise ValueError("Unknown training selection mode")
    if start < 0 or samples == 0 or samples < -1:
        raise ValueError("start must be nonnegative; samples must be positive or -1")
    if not rows or not heldout:
        raise ValueError("Training and held-out sources must both be nonempty")
    forbidden = {question(row) for row in heldout}
    seen, eligible = set(), []
    for index, row in enumerate(rows):
        query = question(row)
        split = str((row.get("extra_info") or {}).get("split", "train")).lower()
        if split not in {"train", "training"}:
            raise ValueError(f"Non-training row at index {index}: {split}")
        if query in forbidden:
            raise ValueError(f"Training/held-out question overlap at index {index}")
        if query not in seen:
            eligible.append(index)
            seen.add(query)
    random.Random(seed).shuffle(eligible)
    if selection == "task_balanced":
        # Round-robin task types, with seeded order inside and across groups.
        # This uses only task metadata, never rewards, solutions or gold paths.
        groups = {}
        for index in eligible:
            if rows[index].get("ability") != "ScienceWorld@real":
                raise ValueError("task_balanced selection requires ScienceWorld rows")
            groups.setdefault(rows[index]["extra_info"]["task_name"], []).append(index)
        names = sorted(groups)
        random.Random(seed).shuffle(names)
        eligible = [groups[name][offset] for offset in range(max(map(len, groups.values())))
                    for name in names if offset < len(groups[name])]
    selected = eligible[start:] if samples == -1 else eligible[start:start + samples]
    if not selected:
        raise ValueError("No training rows selected")
    return selected


def make_config(args):
    protocol = getattr(args, "graph_protocol", "explicit_tools")
    if getattr(args, "benchmark", "bcp") == "scienceworld":
        from scripts.eval_agent_benchmarks import config_for as simulator_config
        config = simulator_config("scienceworld", "contextgraph", args.prompt_length + args.response_length,
                                  args.max_steps, "turns", "focus_v3", graph_action_protocol=protocol)
    else:
        config = config_for(SimpleNamespace(method="contextgraph", memory_mode="repaired",
                                            graph_action_protocol=protocol))
    rollout = config.actor_rollout_ref.rollout
    rollout.prompt_length, rollout.response_length = args.prompt_length, args.response_length
    rollout.plugin.update({"capture_model_contexts": True,
                           "branch_len": args.branch_len, "val_response_length": args.response_length,
                           "max_turn": args.max_turn, "val_max_turn": args.max_turn,
                           "turn_max_new_tokens": args.turn_max_tokens})
    from agents.sft_teacher_guidance import teacher_guidance
    profile = getattr(args, "teacher_profile", "evaluation")
    if protocol == "controller" and profile != "evaluation":
        raise ValueError("Auto collection uses the evaluation controller, not explicit teacher guidance")
    teacher_guidance(profile)  # Validate before starting a teacher or simulator.
    rollout.plugin.sft_teacher_profile = profile
    if protocol == "controller":
        # Make runtime defaults visible in saved configuration, rather than
        # leaving automatic history management implicit in the executor.
        rollout.plugin.enable_retrieval_memory = True
    # Match the chat endpoint's non-thinking template, including budget accounting.
    rollout.plugin.apply_chat_template_kwargs = {"enable_thinking": False}
    return config


class TeacherClient:
    """Chat API adapter retaining actual XML actor / JSON controller requests.

API output IDs below are student re-tokenizations, NOT teacher sampled IDs or
teacher policy log-probabilities. They are only for SFT/inference accounting.
"""
    def __init__(self, args, tokenizer, config, audit_path, client=None):
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=args.teacher_url,
                                 api_key=os.environ.get("DEEPSEEK_API_KEY") or "EMPTY",
                                 timeout=600, max_retries=2)
        self.client, self.args, self.tokenizer, self.config = client, args, tokenizer, config
        self.audit_path = Path(audit_path)

    def request(self, messages, limit, schema, constraints=None):
        body = dict(model=self.args.teacher_model, messages=copy.deepcopy(messages),
                    max_tokens=limit, temperature=self.args.temperature, top_p=0.95)
        if self.args.transport == "deepseek":
            body.update(extra_body={"thinking": {"type": "disabled"}})
            if schema is not None:
                body["response_format"] = {"type": "json_object"}
            if constraints and "regex" in constraints:
                raise ValueError("Regex action constraints require local vLLM")
        else:
            body["extra_body"] = {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
            if constraints or schema is not None:
                body["extra_body"]["structured_outputs"] = constraints or {"json": schema}
        return body

    async def create_completion(self, input_ids, **kwargs):
        limit = completion_budget(self.config, input_ids, kwargs)
        if limit < 256:
            return None
        constraints = kwargs.get("structured_outputs")
        schema = (constraints or {}).get("json")
        request = self.request(kwargs["messages"], limit, schema, constraints)
        for key in ("temperature", "top_p"):
            if key in (kwargs.get("sampling_params") or {}):
                request[key] = kwargs["sampling_params"][key]
        response = await self.client.chat.completions.create(**request)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        # Persist before the framework parses or dispatches, including failures.
        record = {"context_uid": kwargs.get("uid"), "messages": request["messages"],
                  "response": content, "schema": schema, "max_tokens": limit,
                  "finish_reason": choice.finish_reason, "usage": usage,
                  "requested_model": self.args.teacher_model, "returned_model": response.model,
                  "response_id": response.id, "transport": self.args.transport}
        record["structured_outputs"] = constraints
        record["sampling"] = {k: request[k] for k in ("temperature", "top_p")}
        record['sft_assistance'] = kwargs.get('sft_assistance')
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        ids = self.tokenizer.encode(content, add_special_tokens=False)
        return {"choices": [{"message": {"content": content, "raw_output_ids": ids,
                    "response_log_probs": [0.0] * len(ids), "extra_data": {"input_ids": list(input_ids)},
                    "metrics": {"usage": usage}}, "finish_reason": choice.finish_reason}]}

    async def close(self):
        await self.client.close()


async def preflight(args, tokenizer, config, root):
    import httpx
    from agents.action_contract import ActionContract
    scienceworld = getattr(args, "benchmark", "bcp") == "scienceworld"
    if not scienceworld:
        async with httpx.AsyncClient(timeout=30) as http:
            (await http.get(args.search_url.rstrip("/") + "/health")).raise_for_status()
    teacher = TeacherClient(args, tokenizer, config, root / "preflight-requests.jsonl")
    contract = ActionContract("LocalSearch", None)
    try:
        if getattr(args, "graph_protocol", "explicit_tools") == "controller":
            from agents.graph_controller import graph_action_schema
            from agents.agent_text import extract_fn_call
            import jsonschema
            schema = graph_action_schema([0, 1], allow_pass=True)
            probes = [(None, '<function=action><parameter=command>look around</parameter></function>'
                       if scienceworld else '<function=search><parameter=query>test</parameter></function>'),
                      (schema, '{"action":"pass","candidate_indices":[],"summary":"","relation":"semantic"}')]
            for schema, text in probes:
                messages = [{"role": "user", "content": "Return exactly: " + text}]
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
                result = await teacher.create_completion(ids, messages=messages, max_new_tokens=256,
                                                        structured_outputs={"json": schema} if schema else None)
                content = result["choices"][0]["message"]["content"]
                if schema:
                    jsonschema.validate(json.loads(content), schema)
                elif extract_fn_call(content) is None:
                    raise ValueError("Auto teacher preflight did not produce an environment tool call")
            return
        for branch in (False, True):
            contract = ActionContract("ScienceWorld@real" if scienceworld else "LocalSearch", None, branch=branch)
            messages = [{"role": "user", "content": 'Return JSON: {"tool":"graph_read","arguments":{}}'}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False)
            result = await teacher.create_completion(ids, messages=messages, max_new_tokens=256,
                                                      structured_outputs={"json": contract.schema()})
            payload, _ = contract.decode(result["choices"][0]["message"]["content"], contract.schema())
            if payload["tool"] != "graph_read":
                raise ValueError("Teacher preflight did not produce the requested action")
    finally:
        await teacher.close()


async def collect(args, rows, indices, config, tokenizer, root):
    from agents.utils import TaskContext
    from scripts.eval_gaia import _make_dataproto, _metrics_from_output
    from agents.graph_agent_isolated import process_item

    await preflight(args, tokenizer, config.actor_rollout_ref.rollout, root)
    if args.preflight_only:
        return
    semaphore = asyncio.Semaphore(args.workers)
    benchmark = getattr(args, "benchmark", "bcp")
    workflow = "scienceworld_graph" if benchmark == "scienceworld" else "search_graph"

    async def one(index, candidate):
        async with semaphore:
            episode_root = root / "episodes" / f"{index:06d}-{candidate:02d}"
            completed = episode_root / "episode.json"
            if completed.exists():
                print(f"RESUME_SKIP {completed}", flush=True)
                return
            attempt = episode_root / ("attempt-" + uuid.uuid4().hex[:12])
            attempt.mkdir(parents=True)
            client = TeacherClient(args, tokenizer, config.actor_rollout_ref.rollout, attempt / "requests.jsonl")
            row = copy.deepcopy(rows[index])
            extra = row["extra_info"]
            extra["instance_id"] = f"{benchmark}-train-{index}-{candidate}"
            extra["workflow"] = workflow
            extra["tool_log"] = str(attempt / "tools.jsonl")
            item = _make_dataproto(row, workflow)
            item.meta_info["max_turn"] = args.max_turn
            record = {"benchmark": benchmark, "source_index": index, "candidate": candidate,
                      "question_hash": digest(question(row)), "status": "error", "task_reward": 0,
                      "requests": str((attempt / "requests.jsonl").relative_to(root))}
            try:
                context = TaskContext(config=config, global_step=0, llm_client=client,
                                      is_train=False, tokenizer=tokenizer)
                output = await process_item(item, context)
                if not output:
                    raise RuntimeError("No trajectory returned")
                reward, _, finish, trajectory = _metrics_from_output(output)
                (attempt / "trajectory.json").write_text(json.dumps(trajectory, ensure_ascii=False, default=str), encoding="utf-8")
                record.update(status="ok", task_reward=reward, is_finish=finish,
                              trajectory=str((attempt / "trajectory.json").relative_to(root)))
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                # Keep raw failed attempts, never turn them into positive labels.
                (attempt / "error.json").write_text(json.dumps(record), encoding="utf-8")
            finally:
                await client.close()
            if record["status"] == "ok":
                pending = episode_root / "episode.pending.json"
                pending.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                pending.replace(completed)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    await asyncio.gather(*(one(i, c) for i in indices for c in range(args.candidates)))


def checkpoint_provenance(path):
    if not path:
        return None
    root = Path(path).resolve()
    config = root / "config.json"
    if not config.is_file():
        raise ValueError(f"Teacher checkpoint has no config.json: {root}")
    metadata = json.loads(config.read_text(encoding="utf-8"))
    files = [config, *sorted(root.glob("*.index.json"))]
    return {"path": str(root), "model_type": metadata.get("model_type"),
            "architectures": metadata.get("architectures"),
            "metadata_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "weights_hashed": False}


def save_collection_manifest(root, manifest):
    """Keep the collection contract immutable when a local server moves nodes."""
    root = Path(root)
    path = root / 'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        comparable = copy.deepcopy(manifest)
        checkpoint = manifest.get('local_teacher_checkpoint') or {}
        if (manifest.get('settings', {}).get('transport') == 'vllm'
                and checkpoint.get('path') and checkpoint.get('metadata_sha256')):
            comparable['settings']['teacher_url'] = saved['settings'].get('teacher_url')
        if saved != comparable:
            raise ValueError('Resume manifest mismatch; use a fresh output directory')
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Actual transport is recorded separately; old exports keep their source hash.
    with (root / 'collection-sessions.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'session_id': uuid.uuid4().hex,
                                 'teacher_url': manifest['settings']['teacher_url'],
                                 'commit': manifest['commit'],
                                 'source_manifest_sha256': digest(json.loads(path.read_text(encoding='utf-8')))}) + '\n')


def validate_scienceworld_training(rows, official_train):
    allowed = {(t["task_name"], int(t["variation_idx"])) for t in official_train}
    for row in rows:
        extra = row.get("extra_info") or {}
        question(row)
        if (row.get("ability") != "ScienceWorld@real" or extra.get("split") != "train"
                or (extra.get("task_name"), extra.get("variation_idx")) not in allowed
                or extra.get("simplification", "") != ""):
            raise ValueError("ScienceWorld requires official unsimplified training variations")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["bcp", "scienceworld"], default="bcp")
    p.add_argument("--data", default="data/bc_train.parquet")
    p.add_argument("--heldout", default="data/bc_test.parquet")
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    p.add_argument("--teacher-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--teacher-model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
    p.add_argument("--teacher-checkpoint", default=os.getenv("TEACHER_CHECKPOINT_PATH"))
    p.add_argument("--transport", choices=["deepseek", "vllm"], default="deepseek")
    p.add_argument("--search-url", default=os.getenv("LOCAL_SEARCH_URL"))
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--selection", choices=["uniform", "task_balanced"], default="uniform")
    p.add_argument("--teacher-profile", choices=["evaluation", "graph_demonstration_v1", "framework_assisted_v1"], default="evaluation")
    p.add_argument("--graph-protocol", choices=["explicit_tools", "controller"], default="explicit_tools")
    p.add_argument("--prompt-length", type=int, default=8192)
    p.add_argument("--response-length", type=int, default=24576)
    p.add_argument("--branch-len", type=int, default=8192)
    p.add_argument("--max-turn", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--turn-max-tokens", type=int, default=2048)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def main():
    import pandas as pd
    import subprocess
    from omegaconf import OmegaConf
    args = parse_args()
    if min(args.workers, args.candidates, args.prompt_length, args.response_length, args.branch_len, args.max_turn, args.max_steps) < 1 or args.turn_max_tokens < 256:
        raise ValueError("Positive budgets/workers/candidates and turn-max-tokens >= 256 required")
    if any(word in Path(args.data).stem.lower() for word in ("test", "validation", "heldout")):
        raise ValueError("Refusing evaluation split for SFT generation")
    rows = pd.read_parquet(args.data).to_dict("records")
    heldout = pd.read_parquet(args.heldout).to_dict("records")
    simulator = None
    if args.benchmark == "scienceworld":
        import importlib.metadata
        from scripts.make_scienceworld_data import collect_variations
        version = importlib.metadata.version("scienceworld")
        if version != "1.2.3":
            raise ValueError("ScienceWorld 1.2.3 required")
        validate_scienceworld_training(rows, collect_variations("train"))
        simulator = {"version": version, "split": "train", "simplification": "", "prompt_profile": "focus_v3"}
    elif any(row.get("ability") != "LocalSearch" for row in rows + heldout):
        raise ValueError("BC-P requires LocalSearch rows")
    indices = select_training_rows(rows, heldout, args.start, args.samples, args.seed, args.selection)
    config = make_config(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"profile": AUTO_PROFILE if args.graph_protocol == "controller" else (SCIENCEWORLD_PROFILE if args.benchmark == "scienceworld" else PROFILE),
                "simulator": simulator, "source_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "heldout_sha256": hashlib.sha256(Path(args.heldout).read_bytes()).hexdigest(),
                "indices": indices, "settings": {k: v for k, v in vars(args).items()
                                                if k not in {"dry_run", "preflight_only", "output"}},
                "config": OmegaConf.to_container(config),
                "local_teacher_checkpoint": checkpoint_provenance(args.teacher_checkpoint),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "grader": {"kind": "environment"} if simulator else {"model": os.getenv("JUDGE_MODEL", "gpt-5-nano"),
                           "base_url": os.getenv("OPENAI_BASE_URL"),
                           "mode": "llm_fallback" if os.getenv("OPENAI_API_KEY", "") not in {"", "dummy"} else "strict_em_only"},
                "accounting": "student tokenizer for episode budgets; teacher API usage saved per request; no teacher logprobs",
                "schema_enforcement": "per-request constraints; auto actor XML, controller JSON; see requests.jsonl" if args.graph_protocol == "controller" else ("local validation + API JSON mode" if args.transport == "deepseek" else "vllm JSON schema + local validation")}
    save_collection_manifest(root, manifest)
    print(json.dumps({"selected_questions": len(indices), "planned_episodes": len(indices) * args.candidates,
                      "teacher_profile": args.teacher_profile, "selection": args.selection,
                      "graph_protocol": args.graph_protocol,
                      "task_types": sorted({rows[i]["extra_info"]["task_name"] for i in indices}) if simulator else None,
                      "teacher": args.teacher_model, "output": str(root), "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return
    if args.benchmark == "bcp" and not args.search_url:
        raise ValueError("Set LOCAL_SEARCH_URL or --search-url")
    if args.transport == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("Set DEEPSEEK_API_KEY separately from the grader's OPENAI_API_KEY")
    if args.search_url:
        os.environ["LOCAL_SEARCH_URL"] = args.search_url
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    asyncio.run(collect(args, rows, indices, config, tokenizer, root))


if __name__ == "__main__":
    main()
