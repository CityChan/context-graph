import asyncio
from collections import Counter
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.exact_context import checksum, segments_from_requests, tokenizer_signature, validate_agent_context
from scripts import build_route_a_sft as builder, generate_bcp_graph_sft as generator
from scripts.route_a_client import RouteAClient
from tests.test_auto_graph_sft import arguments
from tests.test_session_restart import Tokenizer as BaseTokenizer


class Tokenizer(BaseTokenizer):
    pad_token_id = 0
    special_tokens_map = {"eos_token": "\0"}
    chat_template = "fixture template"

    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return super().decode([i for i in ids if i != 0] if skip_special_tokens else ids)


def record(messages, response, uid="main"):
    tokenizer = Tokenizer()
    return {"input_ids": tokenizer.apply_chat_template(messages, add_generation_prompt=True),
            "output_ids": tokenizer.encode(response) + [0], "messages": messages,
            "response": response, "context_uid": uid}


def example_requests():
    first = record([{"role": "user", "content": "original evidence"}], "<think>reason</think> action 1")
    second = record(first["messages"] + [{"role": "assistant", "content": first["response"]},
                                       {"role": "user", "content": "next observation"}], "action 2")
    third_messages = copy.deepcopy(second["messages"]) + [
        {"role": "assistant", "content": second["response"]}, {"role": "user", "content": "continue"}]
    third_messages[0]["content"] = "archived summary replaces evidence"
    return [first, second, record(third_messages, "action 3")]


def test_segments_keep_sampled_thinking_and_mask_rewritten_history():
    requests = example_requests()
    signature = tokenizer_signature(Tokenizer())
    segments = segments_from_requests(requests, 65536, signature)
    assert len(segments) == 2
    assert [len(s["agent_context"]["spans"]) for s in segments] == [2, 1]
    observed = []
    for segment in segments:
        ids, mask = validate_agent_context(segment, 65536, signature)
        for span in segment["agent_context"]["spans"]:
            sampled = requests[span["request_index"]]
            assert ids[:span["start"]] == sampled["input_ids"]
            observed += ids[span["start"]:span["end"]]
        assert sum(mask) == sum(len(requests[s["request_index"]]["output_ids"]) for s in segment["agent_context"]["spans"])
    assert observed == sum((r["output_ids"] for r in requests), [])
    historical = [m for m in segments[1]["messages"] if m["role"] == "assistant"]
    assert [m["train"] for m in historical] == [False, False, True]


def test_token_or_message_rewrite_or_branch_separates_segments():
    first, second, _ = example_requests()
    for field in ("input_ids", "messages", "context_uid"):
        changed = copy.deepcopy(second)
        if field == "input_ids":
            changed[field][0] += 1
        elif field == "messages":
            changed[field][0]["content"] = "different"
        else:
            changed[field] = "branch"
        assert len(segments_from_requests([first, changed], 65536, "signature")) == 2
    with pytest.raises(ValueError, match="truncation"):
        segments_from_requests([first], 3, "signature")


def test_dataset_native_mask_survives_parquet_without_retokenization(tmp_path):
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    tokenizer = Tokenizer()
    rows = segments_from_requests(example_requests(), 65536, tokenizer_signature(tokenizer))
    for row in rows:
        row.update(sft_profile="route_a_v1", enable_thinking=True, tools=[])
    path = tmp_path / "data.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    for pad in ("right", "no_padding"):
        dataset = MultiTurnSFTDataset(str(path), tokenizer, config={"max_length": 1024, "truncation": "error",
                                                                   "pad_mode": pad, "loss_mask_mode": "agent_context"})
        for index, row in enumerate(rows):
            result = dataset[index]
            size = len(row["agent_context"]["input_ids"])
            assert result["input_ids"][:size].tolist() == row["agent_context"]["input_ids"]
            assert result["loss_mask"][:size].tolist() == row["agent_context"]["loss_mask"]
        dataset.max_length = 3
        with pytest.raises(ValueError, match="truncation"):
            dataset[0]
        dataset.max_length, dataset.truncation = 1024, "right"
        with pytest.raises(ValueError, match="truncation=error"):
            dataset[0]
    dataset.loss_mask_mode = "chatml"
    with pytest.raises(ValueError, match="requires loss_mask_mode"):
        dataset[0]


def test_reject_tampered_conditioning_mask_and_tokenizer():
    row = segments_from_requests(example_requests(), 65536, "sig")[0]
    with pytest.raises(ValueError, match="tokenizer"):
        validate_agent_context(row, 65536, "other")
    bad = copy.deepcopy(row)
    bad["agent_context"]["input_ids"][0] += 1
    bad["agent_context"]["sha256"] = checksum({k: v for k, v in bad["agent_context"].items() if k != "sha256"})
    with pytest.raises(ValueError, match="conditioning"):
        validate_agent_context(bad, 65536, "sig")
    bad = copy.deepcopy(row)
    bad["agent_context"]["loss_mask"][0] = 1
    bad["agent_context"]["sha256"] = checksum({k: v for k, v in bad["agent_context"].items() if k != "sha256"})
    with pytest.raises(ValueError, match="loss mask"):
        validate_agent_context(bad, 65536, "sig")


@pytest.mark.parametrize("benchmark", ["bcp", "scienceworld"])
@pytest.mark.parametrize("edit", [True, False])
def test_native_collection_success_export_and_resume(tmp_path, monkeypatch, benchmark, edit):
    from agents import graph_agent_isolated
    from scripts import route_a_client
    monkeypatch.setenv("AGENT_BUDGET_MODE", "working_context")

    class Env:
        def __init__(self, *args):
            self.stats = Counter()
            self.instance_info = {"problem_statement": "What temperature?"}
            self.is_finish = self.env_fail = False
        async def init_env(self, item):
            pass
        async def run_action(self, response):
            self.stats["action"] += 1
            if self.stats["action"] == 3:
                self.is_finish = True
                self.stats.update(completed=1, environment_natural_end=1, environment_score=100)
                return {"action": "action" if benchmark == "scienceworld" else "finish"}
            return {"observation": "source A: measured 20 C. " * 20}
        async def get_reward(self, *args):
            return "", 1.0, {}
        def close(self):
            pass

    monkeypatch.setattr(graph_agent_isolated, "select_env", lambda *a: Env)
    actors = iter(['<function=search><parameter=query>temperature</parameter></function>',
                   '<function=open_page><parameter=docid>A</parameter></function>',
                   '<function=finish><parameter=answer>20 C</parameter></function>'] if benchmark == "bcp" else
                  ['<function=action><parameter=command>look around</parameter></function>'] * 3)
    sent = []
    tokenizer = Tokenizer()
    async def post(url, json):
        import json as codec
        sent.append(copy.deepcopy(json))
        schema = (json.get("structured_outputs") or {}).get("json")
        response = codec.dumps({"action": "merge" if edit else "pass", "candidate_indices": [0, 1] if edit else [],
                                "summary": "source A: measured 20 C" if edit else "", "relation": "semantic"}) if schema else next(actors)
        ids = tokenizer.encode(response) + [0]
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"model": "test-teacher", "choices": [{"token_ids": ids, "finish_reason": "stop"}]})
    async def close():
        pass
    fake = SimpleNamespace(post=post, aclose=close)
    monkeypatch.setattr(route_a_client, "RouteAClient", lambda *a: RouteAClient(*a, client=fake))
    async def preflight(*args):
        pass
    monkeypatch.setattr(generator, "preflight", preflight)
    args = arguments(benchmark)
    args.collection_mode, args.transport, args.seed, args.tokenizer = "route_a", "vllm_tokens", 42, str(tmp_path)
    config = generator.make_config(args)
    assert config.actor_rollout_ref.rollout.plugin.apply_chat_template_kwargs.enable_thinking
    config.actor_rollout_ref.rollout.plugin.consolidation_interval = 2
    config.actor_rollout_ref.rollout.plugin.controller_allow_pass = True
    root = tmp_path / "run"
    root.mkdir()
    row = {"ability": "LocalSearch", "extra_info": {"query": "What temperature?", "answer": "secret answer", "split": "train"}}
    if benchmark == "scienceworld":
        from scripts.make_scienceworld_data import to_row
        row = to_row({"task_id": "train-boil-0", "task_name": "boil", "variation_idx": 0, "split": "train"}, "scienceworld_graph")
    asyncio.run(generator.collect(args, [row], [0], config, tokenizer, root))
    episode = builder.read_json(root / "episodes/000000-00/episode.json")
    trajectory = builder.read_json(root / episode["trajectory"])
    requests = builder.read_jsonl(root / episode["requests"])
    assert builder.audit_episode(episode, trajectory, requests, tokenizer) == ("graph" if edit else "control")
    assert sent[0]["temperature"] == 0.7 and sent[2]["temperature"] == 0
    assert requests[0]["input_ids"] == sent[0]["prompt"]
    assert all(r["output_ids"][-1] == 0 for r in requests)
    assert all("secret answer" not in str(r["messages"]) for r in requests)
    (root / "manifest.json").write_text(json.dumps({"profile": generator.ROUTE_A_PROFILE, "indices": [0],
                                                  "tokenizer_sha256": tokenizer_signature(tokenizer), "settings": vars(args)}))
    export = SimpleNamespace(run=str(root), output=str(tmp_path / "sft"), tokenizer=args.tokenizer,
                             max_length=65536, control_ratio=0.5, seed=42, validation_fraction=0,
                             smoke=True, require_complete=True)
    report = builder.build(export, tokenizer)
    assert report["counts"]["accepted_episodes"] == 1
    assert report["training_ready"] == edit
    frame = pd.read_parquet(tmp_path / "sft" / ("graph_train.parquet" if edit else "control_train.parquet"))
    assert frame["enable_thinking"].all()
    # Completed candidates must not issue more model requests on resume.
    count = len(sent)
    asyncio.run(generator.collect(args, [row], [0], config, tokenizer, root))
    assert len(sent) == count
    # Never relabel the sole validation success as training.
    monkeypatch.setattr(builder, "split_for", lambda *args: "validation")
    export.output = str(tmp_path / "validation-only")
    report = builder.build(export, tokenizer)
    assert not report["training_ready"] and report["validation_ready"]
    bad = copy.deepcopy(episode)
    bad["task_reward"] = 0
    with pytest.raises(ValueError, match="not_solved"):
        builder.audit_episode(bad, trajectory, requests, tokenizer)
    requests[0]["output_ids"][0] += 1
    with pytest.raises(ValueError, match="token_text"):
        builder.audit_episode(episode, trajectory, requests, tokenizer)


def test_control_ratio_is_capped_by_graph_rows_and_deterministic():
    graph = [{"trajectory_id": f"g{i}", "kind": "graph"} for i in range(5)]
    control = [{"trajectory_id": f"c{i}", "kind": "control"} for i in range(10)]
    mixed = builder.mix_training(graph, control, 0.5, 42)
    assert Counter(r["kind"] for r in mixed) == {"graph": 5, "control": 2}
    assert mixed == builder.mix_training(graph, list(reversed(control)), 0.5, 42)
    assert builder.mix_training([], control, 0.5, 42) == []


@pytest.mark.parametrize("transport", ["vllm", "deepseek"])
def test_route_a_rejects_teacher_retokenizing_transports(transport):
    args = arguments()
    args.collection_mode, args.transport = "route_a", transport
    with pytest.raises(ValueError, match="vllm_tokens"):
        generator.make_config(args)


def test_resume_native_endpoint_can_move_but_not_tokenizer(tmp_path):
    from tests.test_auto_graph_sft_full import contract
    original = contract()
    original["settings"]["transport"] = "vllm_tokens"
    original["tokenizer_sha256"] = "original-tokenizer"
    generator.save_collection_manifest(tmp_path, original)
    resumed = copy.deepcopy(original)
    resumed["settings"]["teacher_url"] = "http://new-node:8000"
    generator.save_collection_manifest(tmp_path, resumed)
    assert builder.read_json(tmp_path / "manifest.json") == original
    resumed["tokenizer_sha256"] = "changed-tokenizer"
    with pytest.raises(ValueError, match="manifest mismatch"):
        generator.save_collection_manifest(tmp_path, resumed)


def test_collection_cli_pins_tokenizer_and_64k_contract_before_gpu(tmp_path, monkeypatch):
    import sys
    from tokenizers import Tokenizer as NativeTokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=NativeTokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
                                        unk_token="<unk>", chat_template="{{ messages }}")
    tokenizer.save_pretrained(checkpoint)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}))
    from tests.test_bcp_graph_sft import row
    data, heldout = tmp_path / "train.parquet", tmp_path / "heldout.parquet"
    pd.DataFrame([row("training task")]).to_parquet(data)
    pd.DataFrame([row("held out task")]).to_parquet(heldout)
    root = tmp_path / "run"
    argv = ["generate", "--collection-mode", "route_a", "--graph-protocol", "controller", "--transport", "vllm_tokens",
            "--teacher-checkpoint", str(checkpoint), "--tokenizer", str(checkpoint), "--teacher-model", "same-policy",
            "--data", str(data), "--heldout", str(heldout), "--output", str(root), "--samples", "-1",
            "--prompt-length", "8192", "--response-length", "57344", "--dry-run"]
    monkeypatch.setenv("AGENT_BUDGET_MODE", "working_context")
    monkeypatch.setattr(sys, "argv", argv)
    generator.main()
    saved = builder.read_json(root / "manifest.json")
    assert saved["profile"] == generator.ROUTE_A_PROFILE
    assert saved["tokenizer_sha256"] == tokenizer_signature(tokenizer)
    config = saved["config"]["actor_rollout_ref"]["rollout"]
    assert config["prompt_length"] + config["response_length"] == 65536
    assert config["plugin"]["apply_chat_template_kwargs"]["enable_thinking"]
    generator.main()  # Immutable contract resumes.
    tokenizer.chat_template = "changed template"
    tokenizer.save_pretrained(checkpoint)
    with pytest.raises(ValueError, match="manifest mismatch"):
        generator.main()
