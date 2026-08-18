import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from agents.prompts_code import create_chat_code
from envs.d3gym_env import D3GymEnv, extract_input_paths
from envs.d3gym_sandbox import D3GymSandbox, parse_d3gym_verdict
from scripts.cache_d3gym_images import canonical_arch, collect_task_ids, image_path, inspect_image_arch
from scripts.make_d3gym_data import extract_expected_outputs, split_by_repository


def test_verdict_parser_prefers_official_tuple():
    assert parse_d3gym_verdict(0, "logs\n(True, 'matched')\n", "") == (1.0, "tuple", "matched")
    assert parse_d3gym_verdict(0, "(False, 'wrong shape')\n", "")[0] == 0.0
    assert parse_d3gym_verdict(1, '{"score": 0.25, "detail": "partial"}\n', "")[0] == 0.25


def test_metadata_helpers_extract_contract_and_paths():
    script = """
from pathlib import Path
def eval():
    a = Path('/task/pred_results/pred_table.csv')
    b = 'pred_results/figures/pred_plot.png'
    return a.exists() and Path(b).exists(), 'checked'
"""
    assert extract_expected_outputs(script) == ["figures/pred_plot.png", "pred_table.csv"]
    instruction = (
        "Read `datasets/folder with spaces/input.csv` and "
        "benchmark/datasets/plain.tsv, then produce a result."
    )
    assert extract_input_paths(instruction) == [
        "benchmark/datasets/plain.tsv",
        "datasets/folder with spaces/input.csv",
    ]


def test_repository_split_has_no_leakage_and_is_deterministic():
    frame = pd.DataFrame(
        {
            "task_id": [f"task_{i}" for i in range(8)],
            "original_repo": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )
    train1, val1 = split_by_repository(frame, seed=42, val_fraction=0.25)
    train2, val2 = split_by_repository(frame, seed=42, val_fraction=0.25)
    assert train1.equals(train2)
    assert val1.equals(val2)
    assert set(train1.original_repo).isdisjoint(set(val1.original_repo))
    assert len(train1) and len(val1)


def test_local_task_image_execution_and_official_eval(tmp_path):
    task = tmp_path / "source_task"
    (task / "datasets").mkdir(parents=True)
    (task / "datasets" / "input.txt").write_text("40", encoding="utf-8")
    (task / "eval_script.py").write_text(
        "from pathlib import Path\n"
        "def eval():\n"
        "    p = Path('pred_results/answer.txt')\n"
        "    return (p.exists() and p.read_text().strip() == '42', 'answer check')\n",
        encoding="utf-8",
    )
    sandbox = D3GymSandbox(
        task_id="task_test",
        workdir=str(tmp_path / "run"),
        task_dir=str(task),
        runtime="local",
    )
    try:
        assert sandbox.list_input_files() == ["datasets/input.txt"]
        result = sandbox.execute(
            "from pathlib import Path; value=int(Path('datasets/input.txt').read_text()); "
            "Path('pred_results/answer.txt').write_text(str(value+2)); print(value)"
        )
        assert result["success"] is True
        assert sandbox.list_output_files() == ["answer.txt"]
        verdict = sandbox.evaluate()
        assert verdict["score"] == 1.0
        assert verdict["rule"] == "tuple"
    finally:
        sandbox.close()


def test_timeout_output_bytes_are_decoded(monkeypatch):
    sandbox = object.__new__(D3GymSandbox)

    def time_out(*args, **kwargs):
        raise __import__("subprocess").TimeoutExpired(
            cmd=args[0], timeout=3, output=b"partial stdout", stderr=b"partial stderr"
        )

    monkeypatch.setattr("envs.d3gym_sandbox.subprocess.run", time_out)
    proc = sandbox._run(["apptainer", "exec"], 3)
    assert proc.returncode == 124
    assert proc.stdout == "partial stdout"
    assert proc.stderr == "partial stderr\n[D3-Gym] timed out after 3s"


def test_d3gym_env_dispatch_prompt_and_reward(tmp_path, monkeypatch):
    task = tmp_path / "source_task"
    (task / "datasets").mkdir(parents=True)
    (task / "datasets" / "number.txt").write_text("7", encoding="utf-8")
    eval_source = (
        "from pathlib import Path\n"
        "def eval():\n"
        "    p=Path('pred_results/result.json')\n"
        "    return (p.exists() and json.loads(p.read_text())['value'] == 7, 'json check')\n"
        "import json\n"
    )
    (task / "eval_script.py").write_text(eval_source, encoding="utf-8")
    monkeypatch.setenv("D3GYM_WORKDIR_ROOT", str(tmp_path / "workdirs"))
    config = SimpleNamespace(plugin=SimpleNamespace(sandbox_timeout=30, eval_timeout=30))
    env = D3GymEnv(config, tokenizer=None, ability="D3Gym")
    item = SimpleNamespace(
        non_tensor_batch={
            "extra_info": {
                "task_id": "task_test",
                "instruction": "Read datasets/number.txt and write pred_results/result.json",
                "runtime": "local",
                "task_dir": str(task),
                "expected_outputs": ["result.json"],
                "input_paths": ["datasets/number.txt"],
                "eval_script": eval_source,
            }
        }
    )

    asyncio.run(env.init_env(item))
    dispatch_source = Path("agents/utils.py").read_text(encoding="utf-8")
    assert "elif 'D3Gym' in ability:" in dispatch_source
    assert "from envs.d3gym_env import D3GymEnv" in dispatch_source
    chat = create_chat_code(env.instruction, "code", env=env)
    combined = "\n".join(message["content"] for message in chat)
    assert "fresh Python process" in combined
    assert "datasets/number.txt" in combined
    assert "pred_results/result.json" in combined

    response = """<function=python_exec>
<parameter=code>import json
from pathlib import Path
value=int(Path('datasets/number.txt').read_text())
Path('pred_results/result.json').write_text(json.dumps({'value': value}))
print(value)</parameter>
</function>"""
    observation = asyncio.run(env.run_action(response))
    assert "result.json" in observation["observation"]
    finish = asyncio.run(env.run_action("<function=finish><parameter=message>done</parameter></function>"))
    assert finish == {"action": "finish"}
    detail, score, metrics = asyncio.run(env.get_reward(item, [], None))
    assert detail == "json check"
    assert score == 1.0
    assert metrics["produced_files"] == 1
    env.close()


def test_d3gym_30b_launcher_wiring():
    wrapper = Path("scripts/run_d3gym_30b_instruct_8node.sh").read_text(encoding="utf-8")
    idev_wrapper = Path("scripts/smoke_d3gym_qwen3_30b_instruct_4node.sh").read_text(encoding="utf-8")
    submit = Path("scripts/submit_d3gym_30b_train_suite.sh").read_text(encoding="utf-8")
    base = Path("scripts/eval_sab_react_30b_instruct_8node_smoke.sh").read_text(encoding="utf-8")
    assert "D3GYM_MODE=train" in wrapper
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-8}" in wrapper
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}" in idev_wrapper
    assert "D3GYM_MODE=smoke" in idev_wrapper
    assert "D3GYM_STRICT_INIT=${D3GYM_STRICT_INIT:-1}" in idev_wrapper
    assert "d3gym_train_${DATA_SUFFIX}.parquet" in wrapper
    assert "react fold ctxgraph" in submit
    assert "data.train_files=$SCIENCE_TRAIN_FILE" in base
    assert "trainer.val_only=$TRAINER_VAL_ONLY" in base
    assert "+actor_rollout_ref.rollout.plugin.eval_timeout=$EVAL_TIMEOUT" in base
    train_entry = Path("scripts/train_sab.py").read_text(encoding="utf-8")
    assert 'if os.environ.get("D3GYM_RUNTIME"):' in train_entry


def test_image_cache_collects_parquet_task_ids(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {"extra_info": [{"task_id": "task_2"}, {"task_id": "task_1"}, {"task_id": "task_2"}]}
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: frame)
    assert collect_task_ids(["tasks.parquet"], "task_3") == ["task_2", "task_1", "task_3"]
    assert image_path(str(tmp_path), "task_1") == tmp_path / "task_1.sif"


def test_image_architecture_aliases_are_normalized():
    assert canonical_arch("x86_64") == "amd64"
    assert canonical_arch("aarch64") == "arm64"
    assert canonical_arch("arm64v8") == "arm64"


def test_image_architecture_is_recovered_from_apptainer_mismatch(monkeypatch, tmp_path):
    failure = SimpleNamespace(
        returncode=255,
        stdout="",
        stderr=(
            "FATAL: image's architecture (amd64) could not run on "
            "the host's (arm64)"
        ),
    )
    monkeypatch.setattr("scripts.cache_d3gym_images.subprocess.run", lambda *args, **kwargs: failure)
    arch, detail = inspect_image_arch("apptainer", tmp_path / "task_29.sif")
    assert arch == "amd64"
    assert "host's (arm64)" in detail
