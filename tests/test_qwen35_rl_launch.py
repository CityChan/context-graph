import ast
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_file("structured_adapter", "agents/structured_outputs.py")
preflight = load_file("rl_preflight", "scripts/preflight_bcp_qwen35_rl.py")


@pytest.mark.parametrize("modern", [False, True])
def test_controller_schema_reaches_correct_vllm_api(modern):
    class Params:
        def __init__(self, json):
            self.json = json

    module = SimpleNamespace(GuidedDecodingParams=Params)
    if modern:
        module.StructuredOutputsParams = Params
    schema = {"json": {"type": "object", "required": ["action"]}}
    kwargs = adapter.build_vllm_structured_sampling_kwargs(schema, module)
    key = "structured_outputs" if modern else "guided_decoding"
    assert set(kwargs) == {key}
    assert kwargs[key].json == schema["json"]
    kwargs[key].json["required"].append("other")
    assert schema["json"]["required"] == ["action"]


def test_preflight_rejects_partial_download(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"first": "a.safetensors", "second": "b.safetensors"}}))
    (tmp_path / "a.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="b.safetensors"):
        preflight.check_checkpoint(tmp_path)
    (tmp_path / "b.safetensors").write_bytes(b"fixture")
    assert preflight.check_checkpoint(tmp_path)["weight_shards"] == 2


def test_preflight_rejects_wrong_model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(ValueError, match="Expected Qwen3.5"):
        preflight.check_checkpoint(tmp_path)


def test_rl_prompt_filter_counts_tokens_with_mapping_default_tokenizer():
    # Execute the text branch's real length function without loading datasets/Ray.
    tree = ast.parse((ROOT / 'verl/utils/dataset/rl_dataset.py').read_text(encoding='utf8'))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == 'doc2len' and 'tokenizer.apply_chat_template' in ast.unparse(node))

    class Tokenizer:
        def apply_chat_template(self, chat, return_dict=True, **kwargs):
            ids = list(range(len(chat[0]['content'])))
            return {'input_ids': ids, 'attention_mask': [1] * len(ids)} if return_dict else ids

    namespace = {'self': SimpleNamespace(apply_chat_template_kwargs={}, tool_schemas=None,
                                         max_prompt_length=10), 'tokenizer': Tokenizer(), 'prompt_key': 'prompt'}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'prompt-filter', 'exec'), namespace)
    assert namespace['doc2len']({'prompt': [{'role': 'user', 'content': 'x' * 20}]}) == 20
    assert namespace['doc2len']({'prompt': [{'role': 'user', 'content': 'short'}]}) == 5


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash unavailable")
@pytest.mark.parametrize("method,expected", [("both", ["contextgraph", "foldagent"]),
                                           ("foldagent", ["foldagent"]),
                                           ("preflight", ["contextgraph"])])
def test_submitter_sends_method_to_separate_batch_jobs(tmp_path, method, expected):
    # Export a shell function so no real Slurm command can be invoked.
    capture = tmp_path / "submissions.txt"
    env = dict(os.environ, SUBMIT_CAPTURE=capture.as_posix(), SUBMIT_METHOD=method)
    command = 'sbatch() { printf "%s\\n" "$*" >> "$SUBMIT_CAPTURE"; echo 12345; }; export -f sbatch; bash scripts/submit_train_bcp_qwen35_9b_50step.sh "$SUBMIT_METHOD"'
    result = subprocess.run([shutil.which("bash"), "-c", command], cwd=ROOT, env=env,
        capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stderr
    submissions = capture.read_text().splitlines()
    assert len(submissions) == len(expected)
    for submission, selected in zip(submissions, expected):
        if method == "preflight":
            assert "--nodes=1" in submission and "--time=00:10:00" in submission
            assert "--job-name=bcp-9b-preflight" in submission
        else:
            assert f"--job-name=bcp-9b-{selected}-50" in submission
        assert submission.endswith(f"scripts/train_bcp_qwen35_9b_50step.sbatch {selected}")


def test_dependency_probe_uses_fresh_interpreter_and_preserves_traceback(tmp_path, monkeypatch):
    import sys
    (tmp_path / "healthy_probe.py").write_text("value = 1\n")
    (tmp_path / "broken_probe.py").write_text("raise ImportError('probe failure sentinel')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert preflight.import_in_fresh_process("healthy_probe")["fresh_process"]
    assert "healthy_probe" not in sys.modules
    with pytest.raises(RuntimeError, match="probe failure sentinel") as error:
        preflight.import_in_fresh_process("broken_probe")
    assert "Traceback" in str(error.value)


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash unavailable")
def test_preflight_failure_is_visible_in_slurm_stderr_and_suite(tmp_path):
    # Stub site commands; never submit a job, load a model or run a trainer.
    setup = tmp_path / "launch.sh"
    setup.write_text("\n".join([
        "#!/bin/bash",
        "scontrol() { echo node1; }",
        "git() { if [ \"$1\" = rev-parse ]; then echo fake-sha; fi; return 0; }",
        "source() { :; }", "conda() { :; }",
        "find() { echo /dev/null; }", "python() { unset LD_PRELOAD; echo simulated-preflight-failure; return 17; }",
        "export -f scontrol git source conda find python",
        f'bash "{(ROOT / "scripts/train_bcp_qwen35_9b_50step.sh").as_posix()}" contextgraph',
    ]) + "\n", encoding="utf8", newline="\n")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               CONDA_PREFIX=tmp_path.as_posix(), SLURM_JOB_ID="fake", SLURM_JOB_NODELIST="node1",
               PREFLIGHT_ONLY="1")
    (tmp_path / "outputs").mkdir()
    result = subprocess.run([shutil.which("bash"), setup.as_posix()], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 17, result.stdout + result.stderr
    assert "BCP_RL_FAILED" in result.stderr
    assert "stage=dependency_preflight exit=17" in result.stderr
    suite = next((tmp_path / "outputs").glob("*/suite.log")).read_text()
    assert "simulated-preflight-failure" in suite
    assert "BCP_RL_FAILED" in next((tmp_path / "outputs").glob("*/failure.log")).read_text()


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash unavailable")
@pytest.mark.parametrize('method', ['contextgraph', 'foldagent'])
@pytest.mark.parametrize('outcome', ['complete', 'missing_shard', 'trainer_failure'])
def test_four_node_smoke_runs_one_step_and_audits_all_ranks(tmp_path, method, outcome):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'data').mkdir()
    for name in ('bc_train.parquet', 'bc_test.parquet'):
        (tmp_path / 'data' / name).write_text('fixture')
    (tmp_path / 'scripts/check_qwen3_observation_tokens.sh').write_text('exit 0\n')
    base = tmp_path / 'scripts' / ('train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh' if method == 'contextgraph'
                                   else 'train_bc_foldagent_8b_paperfaithful_5node_48h.sh')
    base.write_text('''set -eu
env > "$PROJECT_ROOT/captured.env"
printf '%s\\n' "$@" > "$PROJECT_ROOT/captured.args"
if [ "$TEST_OUTCOME" = trainer_failure ]; then exit 17; fi
mkdir -p "$CHECKPOINT_ROOT/global_step_1/actor"
printf 1 > "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
for rank in 0 1 2; do
  for kind in model optim extra_state; do
    if [ "$TEST_OUTCOME" = missing_shard ] && [ "$rank:$kind" = 2:optim ]; then continue; fi
    printf fixture > "$CHECKPOINT_ROOT/global_step_1/actor/${kind}_world_size_3_rank_${rank}.pt"
  done
done
''', encoding='utf8', newline='\n')
    setup = tmp_path / 'run.sh'
    setup.write_text('\n'.join([
        'scontrol() { printf "node1\\nnode2\\nnode3\\nnode4\\n"; }',
        'git() { if [ "$1" = rev-parse ]; then echo fake-sha; fi; return 0; }',
        'source() { :; }', 'conda() { :; }', 'find() { echo /dev/null; }',
        'python() { unset LD_PRELOAD; return 0; }',
        'export -f scontrol git source conda find python',
        f'bash "{(ROOT / "scripts/smoke_bcp_qwen35_9b_4node_idev.sh").as_posix()}" "{method}"',
    ]) + '\n', encoding='utf8', newline='\n')
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               CONDA_PREFIX=tmp_path.as_posix(), SLURM_JOB_ID='fixture', SLURM_JOB_NODELIST='node[1-4]',
               OPENAI_API_KEY='fixture', TEST_OUTCOME=outcome, PREFLIGHT_ONLY='1')
    result = subprocess.run([shutil.which('bash'), setup.as_posix()], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == {'complete': 0, 'missing_shard': 1, 'trainer_failure': 17}[outcome], result.stdout + result.stderr
    assert ('BCP_RL_SMOKE_COMPLETE' in result.stdout) == (outcome == 'complete')
    captured = dict(line.split('=', 1) for line in (tmp_path / 'captured.env').read_text().splitlines() if '=' in line)
    for key, value in dict(EXPECTED_NUM_NODES='4', TOTAL_TRAINING_STEPS='1', TRAIN_BATCH_SIZE='3',
                           ROLLOUT_N='2', PPO_MINI_BATCH_SIZE='3', VAL_BEFORE_TRAIN='False',
                           TEST_FREQ='-1', SAVE_FREQ='1', LORA_RANK='0', CONTEXT_LENGTH='12288').items():
        assert captured[key] == value
    assert captured['EXPERIMENT_NAME'].startswith('smoke_')
    assert 'actor_rollout_ref.rollout.max_model_len=12288' in (tmp_path / 'captured.args').read_text()
