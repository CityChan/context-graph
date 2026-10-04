import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.prepare_gram_data import prepare, sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("mode", ["shared", "local", "explicit", "missing-data", "missing-python"])
def test_bcp_worktree_data_resolution_and_specific_errors(tmp_path, mode):
    primary, worktree = tmp_path / "primary", tmp_path / "worktree"
    (primary / ".git").mkdir(parents=True)
    (primary / "data").mkdir()
    worktree.mkdir()
    shared = primary / "data/bc_test.parquet"
    shared.write_bytes(b"shared-data")
    selected = shared
    if mode == "local":
        (worktree / "data").mkdir()
        selected = worktree / "data/bc_test.parquet"
        selected.write_bytes(b"local-data")
    elif mode in {"explicit", "missing-data"}:
        selected = tmp_path / "explicit.parquet"
        if mode == "explicit":
            selected.write_bytes(b"explicit-data")
    library = tmp_path / "libtorch_global_deps.so"
    library.write_bytes(b"fixture")
    interpreter = tmp_path / "python"
    interpreter.write_text('#!/usr/bin/env bash\nif [[ "$1" == -c ]]; then printf "%s\\n" "$TEST_LIBRARY"; else printf "SELECTED_DATA=%s\\n" "$DATA_PATH"; fi\n', encoding="utf-8", newline="\n")
    interpreter.chmod(0o755)
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('git() { printf "%s\\n" "$TEST_COMMON_DIR"; }\nexport() { [[ "${1:-}" == LD_PRELOAD=* ]] || builtin export "$@"; }\n', encoding="utf-8", newline="\n")
    env = dict(os.environ, PROJECT_ROOT=worktree.as_posix(), SCRATCH=tmp_path.as_posix(),
               SLURM_JOB_ID="123", OPENAI_API_KEY="test-placeholder", BASH_ENV=mocks.as_posix(),
               GRAM_PYTHON=interpreter.as_posix(), TEST_LIBRARY=library.as_posix(),
               TEST_COMMON_DIR=(primary / ".git").as_posix(), GRAM_RUN_DIR=tmp_path.as_posix(),
               GRAM_ACTOR_URL="http://actor", GRAM_HELPER_URL="http://helper")
    env.pop("DATA_PATH", None)
    if mode in {"explicit", "missing-data"}:
        env["DATA_PATH"] = selected.as_posix()
    if mode == "missing-python":
        env["GRAM_PYTHON"] = (tmp_path / "missing-python").as_posix()
    result = subprocess.run([shutil.which("bash"), (ROOT / "scripts/smoke_gram_bcp_qwen35_9b_4node_idev.sh").as_posix(), "_eval"],
                            env=env, capture_output=True, text=True, timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if mode.startswith("missing"):
        assert result.returncode == 2
        expected = "Missing evaluator Python:" if mode == "missing-python" else "Missing or empty BC-P parquet:"
        assert expected in result.stderr
        assert str(selected.name if mode == "missing-data" else "missing-python") in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        # Git Bash realpath may spell C:/... as /c/... on Windows.
        actual = result.stdout.split("SELECTED_DATA=", 1)[1].strip().replace("\\", "/")
        assert actual.endswith(selected.as_posix().split(":", 1)[-1])


def test_bundled_smoke_is_two_real_rows_with_all_context_and_private_labels(tmp_path):
    source = ROOT / "examples/gram/hotpotqa_dev_first2.json"
    provenance = json.loads((source.parent / "source.json").read_text(encoding="utf-8"))
    assert sha256(source) == provenance["fixture_sha256"]
    rows = json.loads(source.read_text(encoding="utf-8"))
    assert [row["_id"] for row in rows] == provenance["task_ids"]
    assert provenance["row_indices"] == [0, 1]
    assert len(rows) == 2 and all(len(row["context"]) == 10 for row in rows)
    manifest = prepare(source, tmp_path / "prepared", "hotpotqa", "validation")
    assert manifest["count"] == 2
    tasks = json.loads((tmp_path / "prepared/tasks.json").read_text(encoding="utf-8"))
    assert all(set(task) == {"task_id", "question", "documents"} for task in tasks)
    assert all(len(task["documents"]) == 10 for task in tasks)


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("preflight_failure", [False, True])
@pytest.mark.parametrize("bcp", [False, True, "file", "missing"])
def test_smoke_pairs_busy_guard_and_output_paths(tmp_path, occupied, preflight_failure, bcp):
    mock_python = tmp_path / "python"
    library = tmp_path / "libtorch_global_deps.so"
    library.write_bytes(b"mock-library")
    mock_python.write_text('#!/usr/bin/env bash\nif [[ "$TEST_FAIL_PREFLIGHT" == 1 && "$*" == *"agents.utils"* ]]; then echo "mock import failure"; exit 1; fi\nif [[ "$*" == *"importlib.util"* ]]; then printf "%s\\n" "$PROJECT_ROOT/libtorch_global_deps.so"; else mkdir -p "$GRAM_RUN_DIR/data"; fi\n', encoding="utf-8", newline="\n")
    mock_python.chmod(0o755)
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('''scontrol() { printf 'node0\\nnode1\\nnode2\\nnode3\\n'; }
export() { [[ "${1:-}" == LD_PRELOAD=* ]] || builtin export "$@"; }
flock() { :; }
git() { [[ "$1" != rev-parse ]] || echo fixture-commit; }
sleep() { command sleep 0.05; }
curl() {
    local url="${@: -1}" node
    node=${url#http://}; node=${node%%:*}
    [[ "$TEST_OCCUPIED" == 1 && "$node" == node2 ]] && return 0
    [[ -e "$PROJECT_ROOT/started-$node" ]]
}
srun() {
    local node previous='' argument
    for argument in "$@"; do
        [[ "$previous" != -w ]] || node=$argument
        previous=$argument
    done
    if [[ "$*" == *"_eval"* ]]; then
        printf '%s\\n' "$*" > "$PROJECT_ROOT/eval-$node.txt"
        return 0
    fi
    touch "$PROJECT_ROOT/started-$node"
    trap 'exit 0' TERM
    while :; do command sleep 0.05; done
}
''', encoding="utf-8", newline="\n")
    data = tmp_path / "bc_test.parquet"
    data.write_bytes(b"fixture")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               GRAM_PYTHON=mock_python.as_posix(), SLURM_JOB_ID="smoke123", SLURM_JOB_NODELIST="fixture",
               DATA_PATH=data.as_posix(), OPENAI_API_KEY="test-placeholder-not-a-real-key",
               BASH_ENV=mocks.as_posix(), TEST_OCCUPIED=str(int(occupied)), TEST_FAIL_PREFLIGHT=str(int(preflight_failure)))
    if bcp in {"file", "missing"}:
        env.pop("OPENAI_API_KEY", None)
        env["WORK"] = env["HOME"] = tmp_path.as_posix()
        if bcp == "file":
            (tmp_path / ".openai_env").write_text('OPENAI_API_KEY=test-file-placeholder\nJUDGE_BASE_URL=https://judge.example/v1\n', encoding="utf-8")
    # Inspect inherited routing in an actual child process without printing its key.
    if bcp:
        env["OPENAI_BASE_URL"] = "http://stale-actor:18000/v1"
        env.pop("JUDGE_BASE_URL", None)
        env["TEST_CREDENTIAL_SOURCE"] = str(bcp)
        with mock_python.open("a", encoding="utf-8") as handle:
            handle.write('if [[ "$TEST_CREDENTIAL_SOURCE" == file ]]; then [[ "$OPENAI_API_KEY" == test-file-placeholder && "$OPENAI_BASE_URL" == https://judge.example/v1 && "$JUDGE_BASE_URL" == https://judge.example/v1 ]] || exit 3; else [[ -n "$OPENAI_API_KEY" && -z "${OPENAI_BASE_URL:-}" ]] || exit 3; fi\n')
    script = "scripts/smoke_gram_bcp_qwen35_9b_4node_idev.sh" if bcp else "scripts/smoke_gram_qwen35_9b_4node_idev.sh"
    result = subprocess.run([shutil.which("bash"), (ROOT / script).as_posix()],
                            env=env, capture_output=True, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert "test-file-placeholder" not in result.stdout + result.stderr
    assert "test-placeholder-not-a-real-key" not in result.stdout + result.stderr
    if bcp == "missing":
        assert result.returncode != 0
        assert 'No judge key found' in result.stderr
        assert not list(tmp_path.glob("started-*"))
        return
    assert result.returncode == (2 if occupied or preflight_failure else 0), result.stdout + result.stderr
    if occupied or preflight_failure:
        assert ("mock import failure" if preflight_failure else "already responds") in result.stdout
        assert not list(tmp_path.glob("started-*"))
        assert not list(tmp_path.glob("eval-*"))
    else:
        if bcp:
            assert not (tmp_path / "eval-node1.txt").exists()
            assert "_eval" in (tmp_path / "eval-node3.txt").read_text()
        else:
            assert "_eval 0 http://node0:18000" in (tmp_path / "eval-node1.txt").read_text()
            assert "_eval 1 http://node2:18000" in (tmp_path / "eval-node3.txt").read_text()
        run = next((tmp_path / "context-graph-gram/runs").iterdir())
        names = ("suite.log", "preparation.log", "search.log", "server-actor.log", "server-memory.log", "evaluator.log") if bcp else ("suite.log", "preparation.log", "server-0.log", "server-1.log", "evaluator-0.log", "evaluator-1.log")
        for name in names:
            assert (run / name).is_file()
