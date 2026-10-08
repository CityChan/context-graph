import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("method", ["foldagent", "contextgraph", "agentfold", "supo"])
def test_four_node_pairs_and_occupied_server_guard(tmp_path, occupied, method):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "context-graph-swe/data/lite").mkdir(parents=True)
    (scripts / "run_swe_lite_arm_subset_idev.sh").write_text(
        'printf "%s %s %s %s %s\\n" "$SWE_METHOD" "$SWE_SERVER_NODE" "$SWE_EVAL_NODE" '
        '"$SWE_SHARD_INDEX" "$SWE_SHARD_COUNT" > "$PROJECT_ROOT/pair-$SWE_SHARD_INDEX.txt"\n', newline="\n")
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('''hostname() { echo node0; }
scontrol() { printf 'node0\\nnode1\\nnode2\\nnode3\\n'; }
flock() { :; }
sleep() { command sleep 0.1; }
curl() {
    local url="${@: -1}" node
    node=${url#http://}; node=${node%%:*}
    [[ "$TEST_OCCUPIED" == 1 && "$node" == node2 ]] && return 0
    [[ -e "$PROJECT_ROOT/started-$node" ]]
}
srun() {
    local node
    while (( $# )); do
        if [[ "$1" == -w ]]; then node=$2; break; fi
        shift
    done
    touch "$PROJECT_ROOT/started-$node"
    trap 'exit 0' TERM
    while :; do command sleep 0.1; done
}
''', newline="\n")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               SWE_AGENT_ENV=tmp_path.as_posix(), SLURM_JOB_ID="test123", SLURM_JOB_NODELIST="fixture",
               SWE_RUN_DIR=(tmp_path / "run").as_posix(), BASH_ENV=mocks.as_posix(), TEST_OCCUPIED=str(int(occupied)))
    proc = subprocess.run([shutil.which("bash"), (root / "scripts/run_swe_lite_arm_4node_idev.sh").as_posix(), method],
                          env=env, capture_output=True, text=True, timeout=20,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert proc.returncode == (2 if occupied else 0), proc.stdout + proc.stderr
    if occupied:
        assert "already responds" in proc.stdout
        assert not list(tmp_path.glob("pair-*.txt"))
    else:
        assert (tmp_path / "pair-0.txt").read_text().strip() == f"{method} node0 node1 0 2"
        assert (tmp_path / "pair-1.txt").read_text().strip() == f"{method} node2 node3 1 2"


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("method", ["contextgraph", "foldagent", "agentfold", "supo", "invalid"])
@pytest.mark.parametrize("missing_env", [False, True])
def test_batch_dispatch_preserves_resume_and_budget(tmp_path, method, missing_env):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_swe_lite_arm_4node_idev.sh").write_text(
        'printf "%s|%s|%s|%s\\n" "$1" "$SWE_CONTEXT_LENGTH" "$SWE_SAMPLES" "$SWE_RUN_DIR" > dispatched.txt\n',
        newline="\n")
    mocks = tmp_path / "mocks.sh"
    mocks.write_text("git() { echo fixture-commit; }\n", newline="\n")
    for name in ["bin/python", "grading/bin/python", "preparation/bin/python"]:
        path = tmp_path / "env" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (missing_env and name == "grading/bin/python"):
            path.write_text("#!/bin/bash\nexit 0\n", newline="\n")
            path.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(),
               SWE_AGENT_ENV=(tmp_path / "env").as_posix(), SLURM_JOB_ID="batch123",
               SWE_CONTEXT_LENGTH="32768", SWE_SAMPLES="2", SWE_RUN_DIR="/scratch/previous-run",
               BASH_ENV=mocks.as_posix())
    proc = subprocess.run([shutil.which("bash"), (root / "scripts/eval_swe_lite_arm_4node.sbatch").as_posix(), method],
                          env=env, capture_output=True, text=True, timeout=20,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    rejected = missing_env or method == "invalid"
    assert proc.returncode == (2 if rejected else 0), proc.stdout + proc.stderr
    output = tmp_path / "dispatched.txt"
    if rejected:
        assert not output.exists()
    else:
        assert output.read_text().strip() == f"{method}|32768|2|/scratch/previous-run"
