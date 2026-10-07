"""Same-state GraphRPO for the read-only LocalSearch environment.

Replay the exact prefix through the ordinary agent loop, then sample one new
maintenance decision and run its real continuation. Only that decision trains.
No simulator cloning, second executor, graph-quality judge, or inference fork.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from uuid import uuid4


BACKEND = "old_policy_continuation"


class ReplayMismatch(RuntimeError):
    """A prefix could not reproduce the original maintenance state."""


class _CheckpointReached(Exception):
    pass


def _hash(value):
    def encode(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Unserializable replay state: {type(obj).__name__}")
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=encode,
        separators=(",", ":"),
    ).encode()).hexdigest()


def leave_one_out(rewards):
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("Continuation credit requires >=2 finite task rewards")
    return [(len(rewards) * r - sum(rewards)) / (len(rewards) - 1) for r in rewards]


class Continuation:
    """Per-episode ordered tape; never patch globals or share mutable results."""

    def __init__(self, client, checkpoint, *, prefix=None, seed=0):
        self.client = client
        self.checkpoint = checkpoint
        self.tape = [] if prefix is None else prefix.tape
        self.replaying = prefix is not None
        self.recording = prefix is None
        self.cursor = 0
        self.seen = 0
        self.state_hash = None if prefix is None else prefix.state_hash
        self.prefix_seconds = 0.0 if prefix is None else prefix.prefix_seconds
        self.seed = seed
        self.live_calls = 0
        self.target_turn = None
        self.target_input_ids = None
        self.decision = None
        self.error = None

    async def call(self, kind, key, invoke):
        if self.replaying:
            if self.cursor >= len(self.tape) or self.tape[self.cursor][:2] != (kind, key):
                self.error = ReplayMismatch(f"Prefix diverged at {kind} call {self.cursor}")
                raise self.error
            result = copy.deepcopy(self.tape[self.cursor][2])
            self.cursor += 1
            return result
        result = await invoke()
        if self.recording:
            self.tape.append((kind, key, copy.deepcopy(result)))
        return result

    async def create_completion(self, input_ids, **kwargs):
        # Request IDs differ on replay; token inputs, schema and budgets must not.
        key = _hash([input_ids, {k: v for k, v in kwargs.items() if k not in {"uid", "messages"}}])

        async def generate():
            request = copy.deepcopy(kwargs)
            if not self.recording:
                params = dict(request.get("sampling_params") or {})
                params["seed"] = (self.seed + self.live_calls) % (2**31)
                request["sampling_params"] = params
                self.live_calls += 1
            return await self.client.create_completion(input_ids, **request)

        return await self.call("model", key, generate)

    def attach(self, env):
        from envs.local_search import LocalSearch

        if type(env) is not LocalSearch:
            raise ValueError("old_policy_continuation currently supports only LocalSearch")
        self.env = env
        post = env.client._post

        async def replayable_post(path, payload):
            return await self.call("search", _hash([path, payload]), lambda: post(path, payload))

        env.client._post = replayable_post

    def at_checkpoint(self, agent, graph, env, iteration, elapsed):
        self.seen += 1
        if self.seen != self.checkpoint:
            return elapsed
        if env.env_fail:
            raise RuntimeError("Cannot fork a failed search environment")
        state = {
            "input_ids": agent.context(),
            "graph": graph.to_dict(include_archives=True),
            "environment": {k: v for k, v in vars(env).items()
                            if k not in {"config", "tokenizer", "client"}},
            "iteration": iteration,
        }
        state_hash = _hash(state)
        if self.recording:
            self.state_hash = state_hash
            self.prefix_seconds = elapsed
            raise _CheckpointReached
        if self.error or self.cursor != len(self.tape) or state_hash != self.state_hash:
            raise ReplayMismatch("Maintenance state differs after prefix replay")
        self.replaying = False
        self.target_turn = len(agent.chat)
        self.target_input_ids = list(agent.context())
        return self.prefix_seconds

    async def capture_decision(self, agent):
        if self.target_turn is None or self.decision is not None:
            return
        if len(agent.chat) - 1 != self.target_turn:
            raise ReplayMismatch("Maintenance completion did not occupy the expected turn")
        data = await agent.get_data()
        indices = data["response_turn_token_indices"].get(self.target_turn, [])
        if not indices:
            raise ReplayMismatch("Maintenance decision has no trainable tokens")
        if len(indices) != sum(agent.token_mask[self.target_turn]):
            raise ReplayMismatch("Maintenance decision was truncated in training data")
        training_prefix = data["prompt_ids"] + data["response_ids"][:indices[0]]
        if training_prefix != self.target_input_ids:
            raise ReplayMismatch("M training prefix differs from its actual model input")
        indices = set(indices)
        data["response_mask"] = [int(i in indices) for i in range(len(data["response_ids"]))]
        self.decision = copy.deepcopy(data)


def _memory_output(output, decision, *, group, candidate, advantage, audit):
    result = copy.deepcopy(output)
    if decision is not None:
        for field in ("prompt_ids", "response_ids", "response_mask", "response_logprobs", "num_turns"):
            setattr(result, field, decision[field])
        result.extra_fields["messages"] = decision["messages"]
    else:
        result.response_mask = [0] * len(result.response_ids)
    mask = result.response_mask
    result.extra_fields.update(
        uid=group, gen_uid=f"{group}:{candidate}", agent_name="main",
        mask_rollout=False,
        process_reward_mask=[0.0] * len(mask),
        graph_decision_mask=list(mask),
        graph_edit_credit_mask=[advantage * m for m in mask],
        graph_rpo_continuation=audit,
    )
    result.extra_fields.setdefault("env_stats", {}).update(
        graph_rpo_continuation_skipped=int(decision is None),
        graph_rpo_continuation_nonzero=int(advantage != 0),
        graph_rpo_continuation_advantage=advantage,
        graph_rpo_continuation_abs_advantage=abs(advantage),
        graph_rpo_decision_tokens=sum(mask),
    )
    return result


async def run_continuation_group(item, context, run_episode):
    """One prefix, K independent decisions, K real continuations, M-only loss."""
    plugin = context.config.actor_rollout_ref.rollout.plugin
    algorithm = context.config.algorithm
    if plugin.get("contextgraph_memory_mode", "legacy") == "foldagent":
        raise ValueError("Continuation training requires ContextGraph memory")
    if not algorithm.get("graphrpo_memory_only", False):
        raise ValueError("old_policy_continuation requires algorithm.graphrpo_memory_only=True")
    if not plugin.get("structured_graph_controller", False):
        raise ValueError("Continuation training requires the structured graph controller")
    if plugin.get("structured_memory_enabled", False) or plugin.get("enable_summary", False):
        raise ValueError("Continuation v1 requires ordinary graph memory without session restarts")
    if float(plugin.get("graph_controller_temperature", 0.0)) <= 0:
        raise ValueError("Continuation training requires graph_controller_temperature > 0")
    count = int(plugin.get("graph_rpo_continuation_samples", 4))
    checkpoint = int(plugin.get("graph_rpo_continuation_checkpoint", 1))
    if count < 2 or checkpoint < 1:
        raise ValueError("Continuation samples must be >=2 and checkpoint must be >=1")

    async def run(session):
        child = copy.copy(context)
        child.llm_client = session
        outputs = await run_episode(item, child, _continuation=session)
        if session.error:
            raise session.error
        main = next(out for out in outputs if out.extra_fields["agent_name"] == "main")
        stats = main.extra_fields.get("env_stats", {})
        if session.env.env_fail or stats.get("call_fail") or stats.get("hit_timeout") or stats.get("judge_parse_failure"):
            raise RuntimeError("Continuation infrastructure/judge failure; discard the whole group")
        return main

    prefix = Continuation(context.llm_client, checkpoint)
    group = uuid4().hex
    try:
        output = await run(prefix)
    except _CheckpointReached:
        pass
    else:
        # Early completion is not a bad memory decision: there was no decision.
        return [_memory_output(output, None, group=group, candidate=0, advantage=0,
                               audit={"backend": BACKEND, "skipped": "checkpoint_not_reached"})]

    candidates = []
    base_seed = int(group[:8], 16) % (2**31)
    for index in range(count):
        session = Continuation(context.llm_client, checkpoint, prefix=prefix, seed=base_seed + index * 100003)
        output = await run(session)
        if session.replaying or session.decision is None:
            raise ReplayMismatch("Continuation did not reach the recorded checkpoint")
        candidates.append((session, output))
    rewards = [float(output.reward_score) for _, output in candidates]
    advantages = leave_one_out(rewards)
    outputs = []
    for index, ((session, output), advantage) in enumerate(zip(candidates, advantages)):
        audit = {
            "backend": BACKEND, "group": group, "checkpoint": checkpoint,
            "state_hash": prefix.state_hash, "global_step": context.global_step,
            "candidate": index, "seed": session.seed, "rewards": rewards,
            "advantage": advantage, "prefix_calls": len(prefix.tape),
            "continuation_model_calls": session.live_calls,
        }
        outputs.append(_memory_output(output, session.decision, group=group,
                                      candidate=index, advantage=advantage, audit=audit))
    return outputs
