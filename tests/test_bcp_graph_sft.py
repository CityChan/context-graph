import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agents.context_graph import ContextGraph, NodeType
from agents.graph_trace import GraphTraceRecorder
from scripts.generate_bcp_graph_sft import TeacherClient, select_training_rows, make_config
from scripts.build_bcp_graph_sft import audit_episode, transcript_segments, split_for, review_edits
from agents.action_contract import ActionContract


def row(query, split="train"):
    return {"ability": "LocalSearch", "extra_info": {"query": query, "answer": "secret answer", "split": split}}


def request(tool, args=None, messages=None, uid="main"):
    return {"response": json.dumps({"tool": tool, "arguments": args or {}}),
            "messages": messages or [{"role": "user", "content": "Find evidence"}],
            "context_uid": uid, "schema": ActionContract("LocalSearch", None).schema(), "finish_reason": "stop"}


def fixture():
    graph = ContextGraph()
    graph.model_owned = True
    graph.add_node("Find evidence", NodeType.QUERY)
    graph.add_node("obsolete", NodeType.OBSERVATION)
    trace = GraphTraceRecorder(graph)
    edit = request("prune", {"node_id": "n2"})
    before = trace.capture(graph)
    graph.prune("n2")
    trace.record(graph, before, source="model", op="prune", args={"node_id": "n2"},
                 turn_id=0, assistant_content=edit["response"], decision_context={"mode": "explicit_tool"})
    episode = {"status": "ok", "task_reward": 1.0, "is_finish": True}
    trajectory = {"graph_action_protocol": "explicit_tools", "env_stats": {"natural_finish": 1},
                  "graph_trace": trace.finalize(graph)}
    return episode, trajectory, [edit, request("finish", {"answer": "answer"})]


def test_train_selection_rejects_test_overlap_before_sampling_and_groups_duplicates():
    with pytest.raises(ValueError, match="overlap"):
        select_training_rows([row("First"), row(" Held   Out ")], [row("held out")], 0, 1, 42)
    with pytest.raises(ValueError, match="Non-training"):
        select_training_rows([row("First", "test")], [row("held out")], 0, 1, 42)
    assert len(select_training_rows([row("First"), row("first"), row("Second")], [row("held out")], 0, -1, 42)) == 2


def test_trace_audit_requires_real_sampled_graph_action_and_successful_continuation():
    episode, trajectory, requests = fixture()
    reason, audit = audit_episode(episode, trajectory, requests)
    assert reason is None and audit["kind"] == "graph"
    bad = copy.deepcopy(requests)
    bad[0] = request("graph_read")
    assert audit_episode(episode, trajectory, bad)[0] == "unmatched_graph_action"
    bad = copy.deepcopy(trajectory)
    bad["graph_trace"]["events"][0]["before_hash"] = "tampered"
    assert audit_episode(episode, bad, requests)[0] == "invalid_graph_trace"
    episode["task_reward"] = 0
    assert audit_episode(episode, trajectory, requests)[0] == "task_not_solved"


def test_forced_finish_or_protocol_error_is_not_accepted_even_with_correct_answer():
    for field in ("forced_finish", "action_protocol_errors", "graph_invalid_ops", "judge_parse_failure"):
        episode, trajectory, requests = fixture()
        trajectory["env_stats"][field] = 1
        assert audit_episode(episode, trajectory, requests)[0] == "unclean_episode"
    episode, trajectory, requests = fixture()
    requests[0]["finish_reason"] = "length"
    assert audit_episode(episode, trajectory, requests)[0] == "incomplete_request"


def test_exact_prefixes_survive_compaction_and_branches_without_duplicate_prefix_rows():
    first = request("search", {"query": "facts"})
    second_messages = first["messages"] + [{"role": "assistant", "content": first["response"]},
                                          {"role": "user", "content": "source A"}]
    second = request("graph_compact", messages=second_messages)
    third = request("finish", {"answer": "A"}, messages=[{"role": "user", "content": "Task plus compacted graph"}])
    branch = request("search", {"query": "branch facts"}, uid="branch")
    segments = transcript_segments([first, second, branch, third])
    assert len(segments) == 3
    assert len(segments[0]) == 4
    assert "source A" not in str(segments[-1])
    changed = copy.deepcopy(second)
    changed["messages"][0]["content"] = "changed past context"
    with pytest.raises(ValueError, match="conditioning prefix"):
        transcript_segments([first, changed])


def test_question_split_does_not_depend_on_candidate_or_episode_order():
    assert split_for("same-question", 42, 0.1) == split_for("same-question", 42, 0.1)
    assert split_for("same-question", 42, 0) == "train"
    assert {split_for("same-question", 42, 0.5) for candidate in range(10)} in ({"train"}, {"validation"})


def test_teacher_transport_does_not_rewrite_raw_json_or_use_judge_credentials(tmp_path, monkeypatch):
    from tests.test_session_restart import make_agent
    agent = make_agent([])
    agent.config.plugin.turn_max_new_tokens = 2048
    captured = []
    async def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=' {"tool":"graph_read","arguments":{}} '), finish_reason="stop")],
                               usage=None, model="deepseek-flash", id="test-response")
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    args = SimpleNamespace(teacher_model="deepseek-flash", transport="deepseek", temperature=0.7)
    client = TeacherClient(args, agent.tokenizer, agent.config, tmp_path / "requests.jsonl", client=fake)
    response = asyncio.run(client.create_completion([], messages=[{"role": "user", "content": "Return JSON"}],
                             max_new_tokens=256, structured_outputs={"json": ActionContract("LocalSearch", None).schema()}))
    assert response["choices"][0]["message"]["content"].startswith(" ")
    assert captured[0]["model"] == "deepseek-flash"
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["extra_body"]["thinking"]["type"] == "disabled"
    assert "secret answer" not in json.dumps(captured)
    assert json.loads((tmp_path / "requests.jsonl").read_text())["returned_model"] == "deepseek-flash"
    args.transport = "vllm"
    assert "structured_outputs" in client.request([], 256, {})["extra_body"]


def test_explicit_config_disables_framework_graph_decisions():
    args = SimpleNamespace(prompt_length=8192, response_length=24576, branch_len=8192, max_turn=100, turn_max_tokens=2048)
    config = make_config(args)
    plugin = config.actor_rollout_ref.rollout.plugin
    assert plugin.graph_action_protocol == "explicit_tools"
    assert not plugin.structured_graph_controller and not plugin.auto_prune_max_active
    assert plugin.capture_model_contexts


def test_evidence_review_rejects_unsupported_or_oversized_edits():
    _, trajectory, _ = fixture()
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1] * 20)
    class Reviewer:
        async def create_completion(self, *args, **kwargs):
            return {"choices": [{"message": {"content": json.dumps({"supported": False,
                   "preserves_needed_evidence": True, "reason": "Unique evidence removed"})}, "finish_reason": "stop"}]}
    events = trajectory["graph_trace"]["events"]
    assert asyncio.run(review_edits(events, Reviewer(), tokenizer, 100)) == "review_rejected"
    assert asyncio.run(review_edits(events, None, tokenizer, 10)) == "review_input_too_long"


def test_real_executor_collects_and_exports_compacted_json_policy(tmp_path, monkeypatch):
    from collections import Counter
    from agents import graph_agent_isolated
    from scripts import generate_bcp_graph_sft as generator, build_bcp_graph_sft as curator
    from tests.test_session_restart import Tokenizer
    from omegaconf import OmegaConf
    import pandas as pd

    class Env:
        def __init__(self, *args):
            self.stats = Counter()
            self.instance_info = {"problem_statement": "What temperature?"}
            self.is_finish = False
        async def init_env(self, item):
            pass
        async def run_action(self, response):
            self.stats["action"] += 1
            if "<function=finish>" in response:
                self.is_finish = True
                return {"action": "finish"}
            return {"observation": "source A: measured 20 C; pressure unknown. " * 10}
        async def get_reward(self, *args):
            return "", 1.0, {}
        def close(self):
            pass

    monkeypatch.setattr(graph_agent_isolated, "select_env", lambda *a: Env)
    responses = iter([request("search", {"query": "temperature"})["response"],
                      request("open_page", {"docid": "A"})["response"],
                      request("merge", {"node_ids": ["n2", "n3"], "summary": "source A: 20 C; pressure unknown"})["response"],
                      request("graph_compact")["response"], request("finish", {"answer": "20 C"})["response"]])
    async def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)), finish_reason="stop")],
                               usage=None, model="test-teacher", id="mock")
    async def close():
        pass
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
    base_client = generator.TeacherClient
    monkeypatch.setattr(generator, "TeacherClient", lambda *a: base_client(*a, client=fake))
    async def preflight(*args):
        pass
    monkeypatch.setattr(generator, "preflight", preflight)
    args = SimpleNamespace(prompt_length=16000, response_length=32000, branch_len=8192, max_turn=30,
                           turn_max_tokens=2048, workers=1, candidates=1, teacher_model="test-teacher",
                           transport="deepseek", temperature=0.7, preflight_only=False)
    config = make_config(args)
    root = tmp_path / "run"
    root.mkdir()
    tokenizer = Tokenizer()
    asyncio.run(generator.collect(args, [row("What temperature?")], [0], config, tokenizer, root))
    episode_path = root / "episodes/000000-00/episode.json"
    assert episode_path.exists()
    episode = curator.read_json(episode_path)
    trajectory = curator.read_json(root / episode["trajectory"])
    requests = curator.read_jsonl(root / episode["requests"])
    reason, audit = audit_episode(episode, trajectory, requests)
    assert reason is None, reason
    assert len(requests) == 5 and audit["kind"] == "graph"
    segments = transcript_segments(requests)
    assert len(segments) == 2
    assert all("secret answer" not in str(r["messages"]) for r in requests)
    (root / "manifest.json").write_text(json.dumps({"profile": generator.PROFILE,
                      "settings": vars(args), "config": OmegaConf.to_container(config)}))
    monkeypatch.setattr(curator, "TeacherClient", lambda *a: base_client(*a, client=fake))
    reviewed = []
    async def review(events, *args):
        reviewed.extend(events)
        return None
    monkeypatch.setattr(curator, "review_edits", review)
    export = SimpleNamespace(run=str(root), output=str(tmp_path / "sft"), review=True,
                             review_max_input=24000, max_length=48000, validation_fraction=0, seed=42)
    asyncio.run(curator.build(export, tokenizer))
    assert reviewed
    graph_rows = pd.read_parquet(tmp_path / "sft/graph_train.parquet")
    assert len(graph_rows) == 2 and not graph_rows.iloc[0]["causal_graph_benefit_verified"]
    assert graph_rows["segment_has_graph_action"].tolist() == [True, False]
    assert not (tmp_path / "sft/control_train.parquet").exists()
    # Completed episode resumes without dispatching any further task requests.
    asyncio.run(generator.collect(args, [row("What temperature?")], [0], config, tokenizer, root))


@pytest.mark.parametrize("transport", ["deepseek", "vllm"])
def test_shell_dry_run_never_starts_services_or_requires_api_key(tmp_path, transport):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    conda = tmp_path / "conda.sh"
    capture = tmp_path / "capture.txt"
    conda.write_text('conda() { :; }\npython() { printf "%s\\n" "$@" > "$CAPTURE"; }\n', encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "CONDA_SH": str(conda), "PROJECT_ROOT": str(repo),
           "SCRATCH": str(tmp_path), "RUN_DIR": str(tmp_path / "raw"),
           "CAPTURE": str(capture), "DRY_RUN": "1", "SAMPLES": "3", "CANDIDATES": "2",
           "DEEPSEEK_API_KEY": "", "TEACHER_TRANSPORT": transport,
           "DEEPSEEK_BASE_URL": "http://localhost:18000/v1", "DEEPSEEK_MODEL": "local-checkpoint"}
    subprocess.run([bash, str(repo / "scripts/generate_bcp_graph_sft_deepseek41.sh")],
                   env=env, check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    arguments = capture.read_text().splitlines()
    assert arguments[0] == "scripts/generate_bcp_graph_sft.py"
    assert "--dry-run" in arguments
    assert arguments[arguments.index("--samples") + 1] == "3"
    assert not (tmp_path / "raw/search.log").exists()


def test_local_checkpoint_provenance_detects_metadata_change(tmp_path):
    from scripts.generate_bcp_graph_sft import checkpoint_provenance
    with pytest.raises(ValueError, match="config.json"):
        checkpoint_provenance(tmp_path)
    config = tmp_path / "config.json"
    config.write_text('{"model_type":"local_deepseek"}')
    first = checkpoint_provenance(tmp_path)
    assert first["model_type"] == "local_deepseek"
    assert first["weights_hashed"] is False
    config.write_text('{"model_type":"different_checkpoint"}')
    assert checkpoint_provenance(tmp_path) != first
    assert checkpoint_provenance(None) is None


def test_local_sbatch_handoff_preserves_judge_and_checkpoint(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh").read_text()
    block = source[source.index('if [ "$BCP_EXPLICIT_GRAPH_SFT" = 1 ]; then', source.index('VLLM_READY=0')):]
    block = block[:block.index('export OPENAI_API_KEY=dummy')]
    script = tmp_path / "handoff.sh"
    script.write_text('set -eu\nbash() { [[ "$TEACHER_TRANSPORT" == vllm && "$DEEPSEEK_MODEL" == actual-model && "$TEACHER_CHECKPOINT_PATH" == /scratch/actual-ckpt && "$OPENAI_API_KEY" == judge-test-value && "$OPENAI_BASE_URL" == https://judge.example/v1 && "$SAMPLES" == 3 && "$WORKERS" == 2 && "$DEEPSEEK_BASE_URL" == http://teacher:18000/v1 && "$LOCAL_SEARCH_URL" == http://retrieval:18999 ]]; }\n' + block)
    env = {**os.environ, "BCP_EXPLICIT_GRAPH_SFT": "1", "TEACHER_HEAD_IP": "teacher",
           "TEACHER_PORT": "18000", "SEARCH_NODE_IP": "retrieval", "SEARCH_PORT": "18999",
           "MODEL_ID": "actual-model", "MODEL_PATH": "/scratch/actual-ckpt",
           "MAX_SAMPLES": "3", "NUM_WORKERS": "2", "ARTIFACT_ROOT": "/scratch/output",
           "PROJECT_ROOT": str(repo), "OPENAI_API_KEY": "judge-test-value",
           "OPENAI_BASE_URL": "https://judge.example/v1", "DEEPSEEK_API_KEY": ""}
    for key in ("SAMPLES", "WORKERS", "RUN_DIR"):
        env.pop(key, None)
    subprocess.run([bash, str(script)], env=env, check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
