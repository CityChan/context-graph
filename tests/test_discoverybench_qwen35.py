import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from scripts import eval_discoverybench_qwen35 as runner


def test_matched_budgets_and_code_workflows():
    left = runner.config_for("contextgraph", 65536).actor_rollout_ref.rollout
    right = runner.config_for("foldagent", 65536).actor_rollout_ref.rollout
    assert left.prompt_length + left.response_length == 65536
    assert left.response_length == right.response_length == 57344
    for key in ("max_turn", "branch_len", "turn_max_new_tokens", "enable_summary", "max_traj"):
        assert left.plugin[key] == right.plugin[key]
    assert left.plugin.workflow == "code_graph"
    assert right.plugin.workflow == "code_branch"
    assert left.plugin.structured_graph_controller and not right.plugin.structured_graph_controller


def test_selection_validates_full_split_before_smoke_sampling(monkeypatch):
    tasks = [{"ability": "DiscoveryBench", "extra_info": {"task_id": f"real:t:{i:03}",
              "dataset_type": "real", "dataset_split": "test", "gold_hypothesis": "gold",
              "metadata": "{}", "input_files": ["data.csv"], "input_rel_paths": ["data.csv"]}}
             for i in reversed(range(239))]
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(tasks))
    selected = runner.load_tasks("fixture", 2)
    assert [t["task_id"] for t in selected] == ["real:t:000", "real:t:001"]
    full = runner.load_tasks("fixture", -1)
    left, right = (runner.shard_tasks(full, i, 2) for i in range(2))
    assert (len(left), len(right)) == (120, 119)
    assert not {t["task_id"] for t in left} & {t["task_id"] for t in right}
    assert sorted(left + right, key=lambda t: t["task_id"]) == full
    assert runner.shard_tasks(selected, 0, 2) == selected[:1]
    assert runner.shard_tasks(selected, 1, 2) == selected[1:]
    tasks[0]["extra_info"]["dataset_split"] = "train"
    with pytest.raises(ValueError, match="real-test"):
        runner.load_tasks("fixture", 2)


def test_empty_smoke_shard_and_invalid_layout(tmp_path):
    assert runner.shard_tasks([{}], 1, 2) == []
    assert runner.summary(tmp_path, [])["mean_hms"] is None
    for index, count in ((0, 0), (-1, 2), (2, 2)):
        with pytest.raises(ValueError, match="shard-count"):
            runner.shard_tasks([], index, count)


def test_summary_never_treats_judge_failure_as_zero(tmp_path):
    ids = ["real:a:m1:q1", "real:a:m1:q2", "real:a:m1:q3"]
    for identity, result in zip(ids, [{"status": "graded", "hms": 0.6, "valid_prediction": True},
                                      {"status": "infrastructure_error", "error": "judge unavailable"}]):
        directory = tmp_path / "instances" / runner.task_key(identity)
        directory.mkdir(parents=True)
        runner.save(directory / "result.json", result)
    value = runner.summary(tmp_path, ids)
    assert value["completed"] == 2 and value["pending"] == 1
    assert value["graded"] == 1 and value["infrastructure_errors"] == 1
    assert value["mean_hms"] is None
    assert value["mean_hms_graded"] == 0.6


def test_missing_output_zero_but_judge_exception_propagates(tmp_path, monkeypatch):
    from envs import discoverybench_eval
    assert runner.grade_prediction({}, tmp_path)["hms"] == 0
    (tmp_path / "pred_results").mkdir()
    runner.save(tmp_path / "pred_results/discovery_result.json", {"hypothesis": "test", "workflow": "analysis"})
    def failed(**kwargs):
        raise RuntimeError("judge unavailable")
    monkeypatch.setattr(discoverybench_eval, "score_hypothesis", failed)
    task = {"query": "q", "gold_hypothesis": "g", "metadata": {}}
    with pytest.raises(RuntimeError, match="judge unavailable"):
        runner.grade_prediction(task, tmp_path)


@pytest.mark.parametrize("method", ["contextgraph", "foldagent"])
def test_generation_routes_code_loop_exports_all_trajectories_and_closes(tmp_path, monkeypatch, method):
    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
    closed = []
    class Client:
        def __init__(self, *args):
            self.failed = False
            self.client = self
        async def aclose(self):
            closed.append(True)
    module("transformers", AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object()))
    module("verl", DataProto=SimpleNamespace)
    module("agents.utils", TaskContext=SimpleNamespace)
    module("scripts.eval_bcp_qwen38", TokenClient=Client, tokenizer_preflight=lambda *a: None)
    async def process(item, context):
        assert item.non_tensor_batch["ability"][0] == "DiscoveryBench"
        assert item.non_tensor_batch["extra_info"][0]["workflow"] == runner.WORKFLOWS[method]
        assert context.is_train is False
        return [SimpleNamespace(extra_fields={"agent_name": name, "messages": [{"role": "assistant", "content": name}],
                                              "env_stats": {"task_reward": 1, "get_final_score": 1, "action": 2},
                                              "graph_rewards": {"task_reward": 1}}) for name in ("main", "branch")]
    module("agents.graph_agent_code_isolated" if method == "contextgraph" else "agents.fold_agent_code", process_item=process)
    args = SimpleNamespace(method=method, context_length=65536, model_path="fixture", endpoint="http://fixture")
    asyncio.run(runner.generate(args, {"task_id": "real:a:m1:q1", "instruction": "analyze"}, tmp_path))
    traces = json.loads((tmp_path / "trajectory.json").read_text())
    assert len(traces) == 2
    assert traces[0]["env_stats"] == {"action": 2}
    assert "graph_rewards" not in traces[0]
    assert closed == [True]


def test_portable_task_folders_do_not_collide():
    assert ":" not in runner.task_key("real:a:m1:q1")
    assert runner.task_key("real:a") != runner.task_key("real/a")


def test_resume_skips_finished_retries_interrupted_and_rejects_drift(tmp_path, monkeypatch):
    fake_lock = ModuleType("fcntl")
    fake_lock.LOCK_EX = fake_lock.LOCK_NB = 1
    fake_lock.flock = lambda *a: None
    monkeypatch.setitem(sys.modules, "fcntl", fake_lock)
    helper = ModuleType("scripts.run_swe_arm_subset")
    calls = []
    def child(argv, log, **kwargs):
        attempt = Path(argv[argv.index("--task") + 1])
        calls.append(attempt)
        if len(calls) == 2:
            raise KeyboardInterrupt("allocation ended")
        runner.save(attempt / "result.json", {"status": "graded", "hms": 0.5, "valid_prediction": True})
    helper.run_command = child
    monkeypatch.setitem(sys.modules, "scripts.run_swe_arm_subset", helper)
    tasks = [{"task_id": f"real:a:m1:q{i}"} for i in range(2)]
    monkeypatch.setattr(runner, "preflight", lambda args: (tasks, {}, {"model": "judge"}))
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "fixture")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "fixture-commit")
    data = tmp_path / "data"
    data.write_text("fixture")
    args = SimpleNamespace(output=tmp_path / "run", data=data, model_path="fixture", context_length=65536,
                           method="contextgraph", task_timeout=120, endpoint="http://fixture",
                           shard_index=0, shard_count=1)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args)
    assert len(calls) == 2
    assert runner.run(args) == 0
    assert len(calls) == 3  # one completed task was skipped
    assert runner.summary(args.output, [t["task_id"] for t in tasks])["mean_hms"] == 0.5
    args.context_length = 32768
    with pytest.raises(ValueError, match="protocol mismatch"):
        runner.run(args)
    args.context_length = 65536
    args.shard_count = 2
    with pytest.raises(ValueError, match="protocol mismatch"):
        runner.run(args)


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("mode", ["both", "contextgraph", "foldagent"])
def test_paired_batch_dispatches_methods_and_shards_on_separate_nodes(tmp_path, mode):
    agent = tmp_path / "env"
    (agent / "bin").mkdir(parents=True)
    (agent / ".ready").touch()
    (agent / "installed.txt").write_text("fixture")
    library = tmp_path / "libfixture.so"
    library.write_text("fixture")
    python = agent / "bin/python"
    python.write_text('#!/bin/bash\nif [[ "$1" == -c ]]; then printf "%s\\n" "$TEST_LIBRARY"; else echo PREFLIGHT; fi\n', newline="\n")
    python.chmod(0o755)
    data = tmp_path / "fixture.parquet"
    data.write_text("fixture")
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('''scontrol() { printf 'node0\\nnode1\\nnode2\\nnode3\\n'; }
export() { if [[ "$1" == LD_PRELOAD=* ]]; then return 0; else builtin export "$@"; fi; }
flock() { :; }
sleep() { command sleep 0.1; }
curl() {
    local node=${@: -1}
    node=${node#http://}; node=${node%%:*}
    [[ -f "$PROJECT_ROOT/started-$node" ]]
}
srun() {
    local node='' method=''
    while (( $# )); do
        if [[ "$1" == -w ]]; then node=$2; fi
        if [[ "$1" == _eval ]]; then method=$2; break; fi
        shift
    done
    if [[ -n "$method" ]]; then
        printf '%s %s %s %s' "$node" "$method" "$5" "$6" > "$PROJECT_ROOT/evaluated-$4"
    else
        touch "$PROJECT_ROOT/started-$node"
        trap 'exit 0' TERM
        while :; do command sleep 0.1; done
    fi
}
''', newline="\n")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               DB_AGENT_ENV=agent.as_posix(), DB_DATA=data.as_posix(),
               DB_RUN_DIR=(tmp_path / "run").as_posix(), SLURM_JOB_ID="fixture", SLURM_JOB_NODELIST="fixture",
               BASH_ENV=mocks.as_posix(), TEST_LIBRARY=library.as_posix(), OPENAI_API_KEY="fixture")
    result = subprocess.run([shutil.which("bash"), (runner.REPO / "scripts/eval_discoverybench_qwen35_9b_4node.sbatch").as_posix(), mode],
                            env=env, capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stdout + result.stderr
    labels = ["contextgraph", "foldagent"] if mode == "both" else [f"{mode}-0", f"{mode}-1"]
    for pair, label in enumerate(labels):
        method = label if mode == "both" else mode
        index, count = (0, 1) if mode == "both" else (pair, 2)
        assert (tmp_path / f"evaluated-{label}").read_text() == f"node{pair * 2 + 1} {method} {index} {count}"
        assert (tmp_path / f"run/evaluator-{label}.log").exists()
        assert (tmp_path / f"run/server-{label}.log").exists()
    assert (tmp_path / "started-node0").exists() and (tmp_path / "started-node2").exists()
