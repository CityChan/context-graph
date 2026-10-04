import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest

from scripts.gram_rl_smoke import audit, prepare_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_real_fixture_smoke_preserves_documents_and_disjoint_roles(tmp_path):
    prepare_smoke(tmp_path, ROOT / "examples/gram/hotpotqa_dev_first2.json")
    train = json.loads((tmp_path / "train/tasks.json").read_text(encoding="utf-8"))
    validation = json.loads((tmp_path / "validation/tasks.json").read_text(encoding="utf-8"))
    assert train[0]["task_id"] != validation[0]["task_id"]
    assert train[0]["question"] != validation[0]["question"]
    assert len(train[0]["documents"]) == len(validation[0]["documents"]) == 10
    assert "answers" not in train[0]
    assert json.loads((tmp_path / "smoke-data.json").read_text())["upstream_split"].endswith("validation")
    with pytest.raises(ValueError, match="fresh"):
        prepare_smoke(tmp_path, ROOT / "examples/gram/hotpotqa_dev_first2.json")


def evidence(root, world_size=3):
    ckpt = root / "checkpoints"
    actor = ckpt / "global_step_2/actor"
    actor.mkdir(parents=True)
    (ckpt / "latest_checkpointed_iteration.txt").write_text("2")
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"fixture")
    (root / "trainer.log").write_text("step:1 - actor/grad_norm:0.25 - actor/kl_loss:0.0\nstep:2 - actor/grad_norm:0.1 - actor/kl_loss:1e-5\n")
    (root / "traces").mkdir()
    (root / "traces/episode.jsonl").write_text('{"kind":"action"}\n{"kind":"memory_call"}\n')


@pytest.mark.parametrize("failure", [None, "shard", "nan", "zero", "helper", "step"])
def test_audit_requires_optimizer_and_helper_evidence(tmp_path, failure):
    evidence(tmp_path)
    if failure == "shard":
        (tmp_path / "checkpoints/global_step_2/actor/optim_world_size_3_rank_2.pt").unlink()
    elif failure in {"nan", "zero", "step"}:
        log = tmp_path / "trainer.log"
        text = log.read_text()
        if failure == "nan":
            text = text.replace("0.25", "nan")
        elif failure == "zero":
            text = text.replace("0.25", "0.0").replace("0.1 ", "0.0 ")
        else:
            text = text.splitlines()[0]
        log.write_text(text)
    elif failure == "helper":
        (tmp_path / "traces/episode.jsonl").write_text('{"kind":"action"}\n')
    if failure:
        with pytest.raises(ValueError):
            audit(tmp_path)
        assert not (tmp_path / "smoke-audit.json").exists()
    else:
        result = audit(tmp_path)
        assert result["steps"] == 2 and not result["checkpoint_reload_verified"]


@pytest.mark.parametrize("roles", [[], [False], [False, True]])
def test_bcp_audit_requires_training_and_validation_search_and_rewards(tmp_path, roles):
    evidence(tmp_path, world_size=2)
    with (tmp_path / "traces/episode.jsonl").open("a") as handle:
        for validation in roles:
            handle.write(json.dumps({"kind": "bcp_reward", "validation": validation,
                                     "external_searches": 1, "task_reward": 0,
                                     "judge_audit": [{"judge_method": "llm_judge"}]}) + "\n")
    if len(roles) == 2:
        assert audit(tmp_path, world_size=2, benchmark="bcp")["bcp_graded_episodes"] == 2
    else:
        with pytest.raises(ValueError, match="training/validation"):
            audit(tmp_path, world_size=2, benchmark="bcp")


@pytest.mark.parametrize("benchmark", ["document-stream", "bcp"])
def test_actual_smoke_overrides_compose_with_training_profile(benchmark):
    from hydra import compose, initialize_config_dir
    base = (ROOT / "scripts/train_gram.sh").read_text()
    base = base[base.index("python -m scripts.train_gram \\"):].replace("\\\n", " ")
    base = re.sub(r'\$\{[A-Z_]+:-([^}]+)\}', r'\1', base)
    replacements = {"PROMPT_LENGTH": "11232", "RESPONSE_LENGTH": "1056", "STEP_TOKENS": "1024",
                    "MODEL_PATH": "model", "GRAM_TRAIN_DATA": "train.parquet", "GRAM_VAL_DATA": "val.parquet",
                    "GRAM_MEMORY_ENDPOINT": "http://frozen:18000", "GRAM_MEMORY_MODEL": "helper",
                    "GRAM_MEMORY_REVISION": "fixed"}
    for key, value in replacements.items():
        base = base.replace("$" + key, value)
    base_args = shlex.split(base.replace('"$@"', ''))[3:]
    smoke = (ROOT / "scripts/smoke_gram_rl_qwen35_9b_4node.sbatch").read_text()
    extra = smoke.split("        args=(", 1)[1].split("\n        )", 1)[0]
    if benchmark == "bcp":
        extra += "\n" + smoke.split("            args+=(", 1)[1].split(")\n", 1)[0]
        extra = extra.replace("$JUDGE_MODEL", "gpt-5-nano")
    extra = extra.replace("$GRAM_RUN_DIR", "/smoke").replace("$GRAM_TRACE_DIR", "/smoke/traces").replace("$SLURM_JOB_ID", "123")
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=base_args + shlex.split(extra))
    from scripts.train_gram import validate_training_config
    validate_training_config(config)
    assert config.actor_rollout_ref.rollout.n == (2 if benchmark == "bcp" else 3)
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == (2 if benchmark == "bcp" else 1)
    if benchmark == "bcp":
        assert config.actor_rollout_ref.rollout.plugin.gram.benchmark == "bcp"
    assert config.actor_rollout_ref.rollout.max_model_len == 12288
    assert config.trainer.total_epochs == config.trainer.test_freq == 2


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("trainer_fails", [False, True])
@pytest.mark.parametrize("benchmark", ["document-stream", "bcp"])
def test_launcher_preserves_busy_nodes_and_propagates_training_failure(tmp_path, busy, trainer_fails, benchmark):
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('''scontrol() { printf 'node0\\nnode1\\nnode2\\nnode3\\n'; }
git() { [[ "$1" != rev-parse ]] || echo test-commit; }
getent() { echo '127.0.0.1 STREAM test-node'; }
sleep() { command sleep 0.05; }
curl() { return 0; }
srun() {
    printf '%s\\n' "$*" >> "$TEST_CALLS"
    case "${*: -1}" in
        _idle) [[ "$TEST_BUSY" == 0 ]]; return ;;
        _prepare) return 0 ;;
        _ready) return 0 ;;
        _train) return "$TEST_TRAIN_RC" ;;
    esac
    while :; do command sleep 0.05; done
}
''', encoding="utf-8", newline="\n")
    calls = tmp_path / "calls.txt"
    env = {**os.environ, "BASH_ENV": mocks.as_posix(), "PROJECT_ROOT": ROOT.as_posix(),
           "SCRATCH": tmp_path.as_posix(), "SLURM_JOB_ID": "123", "SLURM_JOB_NODELIST": "mock",
           "TEST_CALLS": calls.as_posix(), "TEST_BUSY": str(int(busy)),
           "TEST_TRAIN_RC": "7" if trainer_fails else "0"}
    if benchmark == "bcp":
        for name in ("bc_train.parquet", "bc_test.parquet"):
            (tmp_path / name).write_bytes(b"fixture")
        env.update(GRAM_BENCHMARK="bcp", OPENAI_API_KEY="test-only", JUDGE_MODEL="gpt-5-nano",
                   GRAM_BCP_TRAIN_SOURCE=(tmp_path / "bc_train.parquet").as_posix(),
                   GRAM_BCP_VAL_SOURCE=(tmp_path / "bc_test.parquet").as_posix())
    else:
        env["GRAM_BENCHMARK"] = "document-stream"
    proc = subprocess.run([shutil.which("bash"), "scripts/smoke_gram_rl_qwen35_9b_4node.sbatch"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=15,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    invocations = calls.read_text()
    if busy:
        assert proc.returncode != 0
        assert "_prepare" not in invocations and "_train" not in invocations
    else:
        assert proc.returncode == (7 if trainer_fails else 0), proc.stdout + proc.stderr
        assert "_ray-head" in invocations and invocations.count("_ray-worker") == (1 if benchmark == "bcp" else 2)
        if benchmark == "bcp":
            assert "_search" in invocations and "-w node2" in invocations
        assert "_train" in invocations
    assert "ray stop" not in invocations and "scancel" not in invocations
