import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts.generation_audit import (
    combine_degeneration_stats, degeneration_stats, has_degenerate_run,
    require_generation_quality,
)


def requests(path, outputs):
    path.write_text("".join(json.dumps({"input_ids": [0] * 100, "output_ids": ids}) + "\n"
                            for ids in outputs), encoding="utf-8")
    return path


@pytest.mark.parametrize("ids,expected", [
    ([], False), ([0] * 19, False), ([0] * 20, True),
    ([4] + [0] * 21 + [5], True), ([0] * 19 + [1] + [0] * 19, False),
    ([7] * 200, False),
])
def test_consecutive_token_zero_boundary(ids, expected):
    assert has_degenerate_run(ids) is expected


def test_only_output_runs_count_once_per_request(tmp_path):
    first = requests(tmp_path / "a.jsonl", [[0] * 20 + [2] + [0] * 20, [0] * 10])
    second = requests(tmp_path / "b.jsonl", [[0] * 10, [1]])
    stats = degeneration_stats([first, second])
    assert stats["model_requests"] == 4
    assert stats["degenerate_requests"] == 1
    assert stats["degenerate_request_rate"] == 0.25


@pytest.mark.parametrize("total,bad,passed", [(100, 1, True), (200, 2, True), (199, 2, False), (99, 1, False)])
def test_strict_one_percent_threshold(tmp_path, total, bad, passed):
    stats = degeneration_stats([requests(tmp_path / "r.jsonl", [[0] * 20] * bad + [[1]] * (total - bad))])
    assert stats["generation_quality_passed"] is passed
    if passed:
        require_generation_quality(stats)
    else:
        with pytest.raises(RuntimeError, match="exceeds 1%"):
            require_generation_quality(stats)


@pytest.mark.parametrize("line", ['{', '{}', '[]', '{"output_ids": [false]}', '{"output_ids": [-1]}', '{"output_ids": "000"}'])
def test_malformed_audit_is_not_silently_counted_clean(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text(line)
    with pytest.raises(ValueError, match="Invalid generation audit.*:1"):
        degeneration_stats([path])


def test_missing_and_empty_audit_are_not_evidence_of_clean_generation(tmp_path):
    with pytest.raises(FileNotFoundError):
        degeneration_stats([tmp_path / "missing.jsonl"])
    stats = degeneration_stats([])
    assert stats["generation_quality_passed"] is None
    assert stats["degenerate_request_rate"] is None
    combined = combine_degeneration_stats([{}, {"generation_audit": stats}])
    assert combined["generation_unaudited_records"] == 2
    assert combined["generation_quality_passed"] is None


def test_summary_rejects_token_zero_degeneration_after_saving(tmp_path):
    from scripts.eval_bcp_qwen38 import summarize
    manifest = dict(source_sha256="hash", indices=[0, 1, 2], commit="sha", model_path="snapshot",
                    method="foldagent", config={}, seed=42, judge_model="judge")
    for rank in range(3):
        (tmp_path / f"manifest-{rank}.json").write_text(json.dumps(manifest))
        row = dict(source_index=rank, task_reward=1, is_finish=True, status="ok")
        (tmp_path / f"results-{rank}.jsonl").write_text(json.dumps(row) + "\n")
        requests(tmp_path / f"requests-{rank}.jsonl", [[1]])
    # Unselected attempts and preflight probes must not change the denominator.
    requests(tmp_path / "requests-999.jsonl", [[0] * 20])
    requests(tmp_path / "preflight-0.jsonl", [[0] * 20])
    summarize(tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["model_requests"] == 3 and saved["degenerate_requests"] == 0
    requests(tmp_path / "requests-1.jsonl", [[0] * 20])
    with pytest.raises(RuntimeError, match="exceeds 1%"):
        summarize(tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["generation_quality_passed"] is False
    assert saved["degenerate_requests"] == 1
    manifest["server_execution"] = {"requested_enforce_eager": False}
    (tmp_path / "manifest-2.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="server execution mismatch"):
        summarize(tmp_path)


def test_gram_weights_requests_and_marks_legacy_results_unaudited():
    from scripts.eval_gram import summarize
    rows = [{"status": "graded", "benchmark": "bcp", "score": 0,
             "generation_audit": {"model_requests": total, "degenerate_requests": bad}}
            for total, bad in [(1, 1), (99, 0)]]
    result = summarize(rows, 2)
    assert result["degenerate_request_rate"] == 0.01
    assert result["generation_quality_passed"] is True
    rows.append({"status": "infrastructure_error"})
    assert summarize(rows, 3)["generation_quality_passed"] is None


def test_scienceworld_audits_current_results_and_saves_failure(tmp_path):
    from scripts.eval_agent_benchmarks import summary
    from scripts.eval_discoverybench_qwen35 import save, task_key
    directory = tmp_path / "instances" / task_key("a")
    directory.mkdir(parents=True)
    old = directory / "attempt-old"
    old.mkdir()
    requests(old / "requests.jsonl", [[0] * 20])
    row = {"status": "graded", "score": 40, "success": False,
           "generation_audit": {"model_requests": 100, "degenerate_requests": 1}}
    save(directory / "result.json", row)
    assert summary(tmp_path, ["a"], "scienceworld")["generation_quality_passed"] is True
    row["generation_audit"]["degenerate_requests"] = 2
    save(directory / "result.json", row)
    with pytest.raises(RuntimeError, match="exceeds 1%"):
        summary(tmp_path, ["a"], "scienceworld")
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["generation_quality_passed"] is False
    assert saved["model_requests"] == 100


def test_scienceworld_runner_persists_request_counts_and_stops_before_next_task(tmp_path, monkeypatch):
    import scripts.eval_agent_benchmarks as runner
    from scripts.eval_discoverybench_qwen35 import save, task_key
    monkeypatch.setattr(runner, "preflight", lambda _: ([{"task_id": "a"}, {"task_id": "b"}], {"revision": "test"}))
    calls = []
    def command(argv, log, timeout):
        attempt = Path(argv[argv.index("--task") + 1])
        calls.append(attempt)
        requests(attempt / "requests.jsonl", [[0] * 20])
        save(attempt / "result.json", {"status": "graded", "score": 0, "success": False})
    monkeypatch.setattr(runner, "run_command", command)
    args = SimpleNamespace(output=tmp_path, benchmark="scienceworld", method="foldagent", endpoint="unused",
                           model_path="unused", context_length=65536, max_steps=100, task_timeout=30,
                           shard_index=0, shard_count=1, retry_errors=False)
    with pytest.raises(RuntimeError, match="exceeds 1%"):
        runner.run(args)
    assert len(calls) == 1
    saved = json.loads((tmp_path / "instances" / task_key("a") / "result.json").read_text())
    assert saved["generation_audit"]["degenerate_requests"] == 1
    assert json.loads((tmp_path / "summary.json").read_text())["generation_quality_passed"] is False
    # Resume also fails from retained evidence instead of silently running task b.
    with pytest.raises(RuntimeError, match="exceeds 1%"):
        runner.run(args)
    assert len(calls) == 1


@pytest.mark.parametrize("setting", [None, "0", "1", "invalid"])
def test_real_server_launcher_eager_default_and_explicit_optout(tmp_path, setting):
    bash = "C:/Program Files/Git/usr/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).exists():
        pytest.skip("Bash required")
    root = Path(__file__).resolve().parents[1]
    for name in ("cuda/bin/nvcc", "cuda/lib64/libcudart.so", "math/targets/sbsa-linux/include/curand.h", "torch.so"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\nexit 0\n", newline="\n")
        path.chmod(0o755)
    conda = tmp_path / "conda.sh"
    conda.write_text("conda() { :; }\n", newline="\n")
    # Only hardware/dependency checks are mocked. Execute the production shell's
    # argument and environment construction through its final exec statement.
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('python() { echo "$TEST_ROOT/torch.so"; }\n'
                     'mktemp() { echo "$TEST_ROOT/cache"; }\n'
                     'exec() { printf "%s\\n" "$@" > "$TEST_ROOT/argv"; printf "%s" "${TORCHDYNAMO_DISABLE-unset}" > "$TEST_ROOT/dynamo"; exit 0; }\n', newline="\n")
    env = dict(os.environ, PROJECT_ROOT=root.as_posix(), SCRATCH=tmp_path.as_posix(),
               TEST_ROOT=tmp_path.as_posix(), SLURM_JOB_ID="123", CONDA_SH=conda.as_posix(),
               CONDA_PREFIX=tmp_path.as_posix(), BASH_ENV=mocks.as_posix(),
               SERVER_CUDA_HOME=(tmp_path / "cuda").as_posix(), SERVER_CUDA_MATH_ROOT=(tmp_path / "math").as_posix(),
               SERVER_CC="bash", SERVER_CXX="bash", MODEL_PATH="fake-model", BENCHMARK="bcp", TORCHDYNAMO_DISABLE="1")
    env.pop("SERVER_ENFORCE_EAGER", None)
    if setting is not None:
        env["SERVER_ENFORCE_EAGER"] = setting
    result = subprocess.run([bash, (root / "scripts/eval_bcp_qwen38_4node_idev.sh").as_posix(), "_server"],
                            env=env, capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if setting == "invalid":
        assert result.returncode == 2
        assert not (tmp_path / "argv").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        argv = (tmp_path / "argv").read_text().splitlines()
        assert ("--enforce-eager" in argv) is (setting != "0")
        assert (tmp_path / "dynamo").read_text() == ("unset" if setting == "0" else "1")
