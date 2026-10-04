import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.gram_agent import GramConfig, MemoryBackend, MemoryBackendError, make_policy, run_episode, score_episode
from agents.gram_memory import DocumentStream, GraphMemory, parse_action, token_f1
from scripts.eval_gram import summarize
from scripts.prepare_gram_data import convert, prepare


TASK = {"task_id": "x", "question": "Where was the author of Book born?", "documents": [
    {"id": "d1", "title": "Book", "text": "Book was written by Alice."},
    {"id": "d2", "title": "Alice", "text": "Alice was born in Paris."}]}


@pytest.mark.parametrize("text", ["hello <answer>A</answer>", "<answer>A</answer><answer>B</answer>",
                                  "<memory_search/>", "<answer><b>A</b></answer>",
                                  "<answer a='b'>A</answer>", "<think>x</think>", "</think><answer>A</answer>"])
def test_malformed_actions(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_xml_and_f1():
    assert parse_action("<think>x</think><answer>A &amp; B</answer>") == ("answer", "A & B")
    assert token_f1("The Paris!", ["London", "Paris"]) == 1
    assert token_f1("a red red apple", ["red apple"]) == pytest.approx(.8)
    assert token_f1("wrong", ["right"]) == 0


def test_graph_atomic_update_provenance_and_directed_paths():
    g = GraphMemory()
    g.apply([["Book", "Author", "Alice"], ["Alice", "born_in", "London"]], [], "d1")
    g.apply([["Alice", "born_in", "Paris"]], [["Alice", "born_in", "London"]], "d2")
    g.apply([["book", "author", "ALICE"]], [], "d3")
    assert g.edges[("Book", "author", "Alice")] == {"d1", "d3"}
    assert [["Book", "author", "Alice"], ["Alice", "born_in", "Paris"]] in g.search("Book", 2, 1)
    assert all(len(p) == 1 for p in g.search("Book", 1, 3))
    assert g.search("nonexistent", 2, 3) == []
    assert g.observation()["edge_count"] == 2
    assert "triple" not in g.observation()  # Search must expose links, not an always-visible full graph.
    old = copy.deepcopy(g.snapshot())
    with pytest.raises(ValueError):
        g.apply([["good", "r", "x"]], [["absent", "r", "x"]], "d4")
    assert old == g.snapshot()


class Helper:
    def __init__(self):
        self.calls = []

    async def edit(self, operation, content, document, question, graph, threshold):
        self.calls.append((operation, document["id"]))
        row = ["Book", "author", "Alice"] if document["id"] == "d1" else ["Alice", "born_in", "Paris"]
        graph.apply([row], [], document["id"])
        return {"add": [row], "remove": []}


class Policy:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    async def __call__(self, messages, limit):
        self.calls.append(copy.deepcopy(messages))
        return next(self.responses), {"response_ids": [1, 2], "response_mask": [1, 1]}


def test_stream_search_invalid_action_and_answer_isolation():
    policy = Policy(["<memory_insert>Book by Alice</memory_insert>", "invalid",
                     "<memory_search>Book</memory_search>", "<memory_update>Alice born Paris</memory_update>",
                     "<answer>Paris</answer>"])
    events, helper = [], Helper()
    result = asyncio.run(run_episode(TASK, policy, helper, GramConfig(), events.append))
    assert helper.calls == [("memory_insert", "d1"), ("memory_update", "d2")]
    assert [json.loads(m[-1]["content"])["memory_obs"]["documents_consumed"] for m in policy.calls] == [0, 1, 1, 1, 2]
    assert "Alice was born in Paris" not in str(policy.calls[0])  # Future document withheld.
    assert json.loads(policy.calls[-1][-1]["content"])["document_obs"] is None
    assert result["format_reward"] == .8
    assert score_episode(result, ["Paris"])["training_reward"] == pytest.approx(1.08)
    assert result["termination_reason"] == "answer"
    assert len(events) == 5


def test_budget_and_no_answer_cannot_receive_answer_reward():
    result = asyncio.run(run_episode(TASK, Policy(["<memory_search>Book</memory_search>"]*3),
                                    Helper(), GramConfig(max_episode_tokens=12)))
    assert result["steps"] == 2 and result["termination_reason"] == "token_limit"
    assert score_episode(result, ["Paris"])["answer_f1"] == 0
    with pytest.raises(ValueError):
        asyncio.run(run_episode({**TASK, "answers": ["SECRET"]}, Policy([]), Helper(), GramConfig()))


def test_memory_helper_validation_and_cosine_merge():
    backend = MemoryBackend("http://unused", "frozen")
    async def run():
        try:
            empty = GraphMemory()
            await backend.edit("memory_insert", "None", TASK["documents"][0], "q", empty, .9)
            assert not empty.edges  # Deterministic paper skip needs no model/API call.
            backend.embedding_model = "embeddings"
            backend.embedding_cache = {"NYC": [1., 0.], "New York": [.999, .001], "Paris": [0., 1.]}
            aliases = await backend.aliases(["New York", "Paris"], ["NYC"], .9)
            assert aliases == {"New York": "NYC", "Paris": "Paris"}
            responses = iter([["Alice", "Paris"], [["Alice", "born_in", "Unverified"]]])
            async def fake(*args):
                return next(responses)
            backend.json_call = fake
            graph = GraphMemory()
            with pytest.raises(MemoryBackendError, match="unverified"):
                await backend.edit("memory_insert", "x", TASK["documents"][0], "q", graph, .9)
            assert not graph.edges
        finally:
            await backend.aclose()
    asyncio.run(run())


def test_helper_json_failure_and_transport_cleanup():
    import httpx
    backend = MemoryBackend("http://memory", "frozen")
    async def run():
        await backend.client.aclose()
        calls = []
        def reply(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "not JSON"}, "finish_reason": "stop"}]})
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        try:
            with pytest.raises(MemoryBackendError):
                await backend.json_call("entities", "extract", {"document": "x"})
            assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
        finally:
            await backend.aclose()
        assert backend.client.is_closed
    asyncio.run(run())


def test_dataset_strips_labels_preserves_distractors_and_manifest(tmp_path):
    row = {"_id": "1", "question": "q", "answer": "secret", "supporting_facts": [["a", 0]],
           "context": [["distractor", ["d"]], ["a", ["evidence"]]]}
    public, answers = convert(row, "hotpotqa")
    assert "secret" not in json.dumps(public) and answers == ["secret"]
    assert public["documents"][0]["title"] == "distractor"
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([row]), encoding="utf-8")
    result = prepare(source, tmp_path / "ready", "hotpotqa", "validation", parquet=True)
    assert result["count"] == 1 and not result["paper_exact_split"]
    import pandas as pd
    frame = pd.read_parquet(tmp_path / "ready/data.parquet")
    assert json.loads(frame.iloc[0].extra_info["gram_task_json"]) == public
    assert "secret" not in str(frame.iloc[0].prompt)
    with pytest.raises(ValueError, match="immutable"):
        prepare(source, tmp_path / "ready", "hotpotqa", "validation")


def test_musique_and_triviaqa(tmp_path):
    row = {"id": "m", "question": "q", "answer": "a", "answer_aliases": ["b"],
           "paragraphs": [{"title": "x", "paragraph_text": "body", "is_supporting": True}]}
    task, answers = convert(row, "musique")
    assert answers == ["a", "b"] and "is_supporting" not in str(task)
    (tmp_path / "wikipedia").mkdir()
    (tmp_path / "wikipedia/x.txt").write_text("evidence", encoding="utf-8")
    trivia = {"QuestionId": "t", "Question": "q", "Answer": {"Value": "a", "Aliases": ["b"]},
              "EntityPages": [{"Title": "x", "Filename": "x.txt"}]}
    task, answers = convert(trivia, "triviaqa", tmp_path)
    assert task["documents"][0]["text"] == "evidence" and answers == ["a", "b"]
    trivia["EntityPages"][0]["Filename"] = "../../secret"
    with pytest.raises(ValueError, match="escapes"):
        convert(trivia, "triviaqa", tmp_path)


def test_summary_does_not_conflate_infrastructure_and_zero_score():
    rows = [{"status": "graded", "answer_f1": 0., "termination_reason": "max_steps"},
            {"status": "infrastructure_error"}]
    assert summarize(rows, 3) == {"selected": 3, "completed": 2, "pending": 1, "graded": 1,
                                  "infrastructure_errors": 1, "mean_answer_f1_graded": 0.,
                                  "mean_answer_f1": None, "answered": 0}


def test_real_agent_export_and_training_reward_no_label_leak():
    from omegaconf import OmegaConf
    from tests.test_session_restart import Tokenizer, Client
    from scripts.train_gram import process_item
    rollout = {"prompt_length": 12000, "response_length": 1024, "plugin": {
        "enable_summary": False, "gram": {"memory_endpoint": "http://frozen", "memory_model": "helper",
        "memory_revision": "fixed-sha", "frozen_memory_acknowledged": True,
        "episode": {"max_steps": 4, "max_step_tokens": 512}}}}
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": rollout}, "algorithm": {
        "adv_estimator": "foldgrpo", "foldgrpo_process_reward_mode": "relative_extrema",
        "fix_bad_positive_adv": False, "use_kl_in_reward": False}})
    client = Client(["<memory_insert>Book by Alice</memory_insert>",
                     "<memory_insert>Alice born Paris</memory_insert>", "<answer>Paris</answer>"])
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), llm_client=client)
    fields = {"extra_info": {"gram_task_json": json.dumps(TASK)}, "uid": "q", "gen_uid": "episode",
              "reward_model": {"ground_truth": json.dumps(["Paris", "SECRET_GOLD_ALIAS"])}}
    outputs = asyncio.run(process_item(fields, context, memory=Helper()))
    assert len(outputs) == 3
    for output, (prefix, kwargs) in zip(outputs, client.calls):
        assert output.reward_score == pytest.approx(1.1)
        assert output.extra_fields["gen_uid"] == "episode"
        assert not any(output.extra_fields["process_reward_mask"])
        assert "SECRET_GOLD_ALIAS" not in str(kwargs["messages"])
        sequence = output.prompt_ids + output.response_ids
        assert sequence[:len(prefix)] == prefix
        assert all(logp == -.125 for logp, mask in zip(output.response_logprobs, output.response_mask) if mask)


def test_prompt_overflow_fails_before_agent_silent_truncation():
    from omegaconf import OmegaConf
    from tests.test_session_restart import Tokenizer, Client
    client = Client([])
    config = OmegaConf.create({"prompt_length": 20, "response_length": 128, "plugin": {"enable_summary": False}})
    with pytest.raises(MemoryBackendError, match="prompt exceeds"):
        asyncio.run(make_policy(client, Tokenizer(), config)([{"role": "user", "content": "x"*100}], 64))
    assert not client.calls


def test_grpo_deduplicates_segments_and_keeps_reward_above_one():
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_foldgrpo_advantage
    # A long episode appears twice, but its reward must enter the group mean once.
    rewards = torch.tensor([[1.1], [1.1], [.1], [.6], [.4], [.8]])
    episode_ids = np.array(["a", "a", "b", "c", "d", "e"])
    advantage, _ = compute_foldgrpo_advantage(
        rewards, torch.ones_like(rewards), np.array(["q"]*6), episode_ids,
        process_reward_mask=torch.zeros_like(rewards), config={"foldgrpo_process_reward_mode": "relative_extrema"})
    unique = torch.tensor([1.1, .1, .6, .4, .8])
    expected = (1.1-unique.mean())/(unique.std()+1e-6)
    assert advantage[0, 0] == pytest.approx(expected.item())
    assert advantage[0, 0] == advantage[1, 0]


@pytest.mark.parametrize("bcp", [False, True])
def test_evaluator_http_end_to_end_and_resume(tmp_path, monkeypatch, bcp):
    import httpx
    from transformers import AutoTokenizer
    from tests.test_session_restart import Tokenizer
    from scripts.eval_gram import evaluate
    class APITokenizer(Tokenizer):
        def decode(self, ids, skip_special_tokens=False, **kwargs):
            return super().decode([i for i in ids if i or not skip_special_tokens])
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([{**TASK, "answers": ["Paris"]}]), encoding="utf-8")
    data = tmp_path / "data"
    prepare(raw, data, "canonical", "validation")
    if bcp:
        import pandas as pd
        data = tmp_path / "bcp.parquet"
        pd.DataFrame([{"extra_info": {"query": TASK["question"], "answer": "Paris"}}]).to_parquet(data)
        monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-real-key")
        monkeypatch.setenv("LOCAL_SEARCH_URL", "http://corpus")
    output = tmp_path / "run"
    output.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    responses = iter((["<search>Book</search>", "<memory_insert>Book by Alice</memory_insert>",
                       "<open_page>d2</open_page>", "<memory_insert>Alice born Paris</memory_insert>",
                       "<answer>Paris</answer>"] if bcp else
                      ["<memory_insert>Book by Alice</memory_insert>",
                       "<memory_insert>Alice born Paris</memory_insert>", "<answer>Paris</answer>"]))
    helpers = iter([["Book", "Alice"], [["Book", "author", "Alice"]],
                    ["Alice", "Paris"], [["Alice", "born_in", "Paris"]]])
    calls = []
    def reply(request):
        calls.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model", "max_model_len": 20000}]})
        if request.url.path in {"/search", "/open"}:
            doc = TASK["documents"][int(request.url.path == "/open")]
            return httpx.Response(200, json={"results": [{"docid": doc["id"], "url": "http://example.org", "text": doc["text"]}]})
        if request.url.path == "/v1/completions":
            content = next(responses)
            body = json.loads(request.content)
            assert body["max_tokens"] <= 512
            if bcp:
                import re
                assert re.fullmatch(body["structured_outputs"]["regex"], content)
                assert not re.fullmatch(body["structured_outputs"]["regex"], "I must search.\n" + content)
            else:
                assert "structured_outputs" not in body
            return httpx.Response(200, json={"choices": [{"token_ids": list(map(ord, content))+[0], "finish_reason": "stop"}]})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(helpers))}, "finish_reason": "stop"}]})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(**kw, transport=httpx.MockTransport(reply)))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: APITokenizer())
    args = SimpleNamespace(benchmark="bcp" if bcp else "document-stream", data=data, output=output, model_path=model_path, samples=-1, shard_index=0, shard_count=1,
                           max_steps=10, episode_tokens=10000, step_tokens=512, timeout=30, search_hops=2,
                           search_top_k=12, entity_threshold=.9, model="model", model_revision="fixed",
                           memory_model="model", memory_revision="fixed", embedding_model=None,
                           embedding_endpoint=None, embedding_revision=None, context_length=20000, seed=42,
                           endpoint="http://actor", memory_endpoint="http://memory", retry_errors=False,
                           action_decoding="xml_regex" if bcp else "unconstrained",
                           bcp_progress_limit=2 if bcp else 0)
    assert asyncio.run(evaluate(args)) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["accuracy" if bcp else "mean_answer_f1"] == 1 and summary["graded"] == 1
    if bcp:
        assert summary["answered"] == 1 and summary["blank_predictions"] == 0
        assert summary["documents_consumed"] == 2 and summary["mean_format_reward_graded"] == 1
    assert len(list(output.glob("instances/*/attempt-*/segments.json"))) == 1
    assert asyncio.run(evaluate(args)) == 0  # No actor/helper calls on resume.
    assert calls.count("/v1/completions") == (5 if bcp else 3)
    args.action_decoding = "unconstrained" if bcp else "xml_regex"
    args.bcp_progress_limit = 0
    with pytest.raises(ValueError, match="protocol mismatch"):
        asyncio.run(evaluate(args))
    args.action_decoding = "xml_regex" if bcp else "unconstrained"
    args.bcp_progress_limit = 2 if bcp else 0
    args.seed = 43
    with pytest.raises(ValueError, match="protocol mismatch"):
        asyncio.run(evaluate(args))


def test_training_data_preflight_rejects_overlap(tmp_path):
    from scripts.train_gram import check_data
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([{**TASK, "answers": ["Paris"]}]), encoding="utf-8")
    prepare(raw, tmp_path / "train", "canonical", "train", parquet=True)
    prepare(raw, tmp_path / "val", "canonical", "validation", parquet=True)
    with pytest.raises(ValueError, match="overlap"):
        check_data(tmp_path / "train/data.parquet", tmp_path / "val/data.parquet")
    raw.write_text(json.dumps([{**TASK, "task_id": "other", "question": "Different question?", "answers": ["Paris"]}]), encoding="utf-8")
    prepare(raw, tmp_path / "val-disjoint", "canonical", "validation", parquet=True)
    check_data(tmp_path / "train/data.parquet", tmp_path / "val-disjoint/data.parquet")


def test_hydra_gram_training_profile_composes_without_gpu():
    from hydra import compose, initialize_config_dir
    import re
    import shlex
    script = (Path(__file__).resolve().parents[1] / "scripts/train_gram.sh").read_text()
    # Resolve the launcher's default shell substitutions, then compose its actual overrides.
    command = script[script.index("python -m scripts.train_gram \\"):].replace("\\\n", " ")
    command = re.sub(r'\$\{[A-Z_]+:-([^}]+)\}', r'\1', command)
    replacements = {"PROMPT_LENGTH": "30688", "RESPONSE_LENGTH": "2080", "STEP_TOKENS": "2048",
                    "MODEL_PATH": "model", "GRAM_TRAIN_DATA": "train.parquet", "GRAM_VAL_DATA": "val.parquet",
                    "GRAM_MEMORY_ENDPOINT": "http://frozen", "GRAM_MEMORY_MODEL": "helper", "GRAM_MEMORY_REVISION": "fixed"}
    for key, value in replacements.items():
        command = command.replace("$"+key, value)
    args = shlex.split(command.replace('"$@"', ''))[3:]
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=args)
    from scripts.train_gram import validate_training_config
    validate_training_config(config)
    assert config.actor_rollout_ref.rollout.n == 5
    assert config.actor_rollout_ref.actor.optim.lr == 1e-6
    assert config.actor_rollout_ref.actor.kl_loss_coef == .001
