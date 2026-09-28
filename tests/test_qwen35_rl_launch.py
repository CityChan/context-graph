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


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash unavailable")
@pytest.mark.parametrize("method,expected", [("both", ["contextgraph", "foldagent"]),
                                           ("foldagent", ["foldagent"])])
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
        assert f"--job-name=bcp-9b-{selected}-50" in submission
        assert submission.endswith(f"scripts/train_bcp_qwen35_9b_50step.sbatch {selected}")
