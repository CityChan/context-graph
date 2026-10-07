"""Exercise the login-node submitter with a fake sbatch; no cluster jobs are created."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = "submit_discoveryworld_qwen35_9b_all200.sh"
RUNNER = "eval_discoveryworld_qwen35_9b_4node.sbatch"


def execute(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, timeout=30,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, **kwargs)


@pytest.mark.parametrize("fail_second", [False, True])
def test_full_submission_and_partial_failure(tmp_path, fail_second):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).exists() or not shutil.which("git"):
        pytest.skip("Git and Bash required")
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in (SUBMIT, RUNNER):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    for args in (["init", "-q"], ["add", "scripts"],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"]):
        result = execute(["git", "-C", str(repo), *args])
        assert result.returncode == 0, result.stderr
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "capture"
    capture.mkdir()
    stub = fake_bin / "sbatch"
    stub.write_text("""#!/bin/bash
set -eu
method=${!#}
printf '%s\\n' "$@" > "$CAPTURE/$method.args"
printf '%s\\n' "$BENCH_DIFFICULTY" "$BENCH_SAMPLES" "$BENCH_MAX_STEPS" "$BENCH_CONTEXT_LENGTH" "$BENCH_DISCOVERYWORLD_OBSERVATION_PROFILE" "$BENCH_MEMORY_PROFILE" "$SERVER_ENFORCE_EAGER" "${BENCH_DATA-unset}" "${BENCH_AGENT_ENV-unset}" "${BENCH_RETRY_ERRORS-unset}" "$BENCH_RUN_DIR" "$PROJECT_ROOT" > "$CAPTURE/$method.env"
if [[ "$method" == foldagent && "$FAIL_SECOND" == 1 ]]; then exit 7; fi
if [[ "$method" == contextgraph ]]; then echo 65001; else echo '65002;vista'; fi
""", encoding="utf8", newline="\n")
    stub.chmod(0o755)
    env = dict(os.environ, SCRATCH=(tmp_path / "scratch").as_posix(), CAPTURE=capture.as_posix(),
               FAIL_SECOND=str(int(fail_second)), BENCH_DIFFICULTY="Normal", BENCH_SAMPLES="2",
               BENCH_MAX_STEPS="100", BENCH_RUN_DIR="stale", BENCH_DATA="stale",
               BENCH_AGENT_ENV="stale", BENCH_RETRY_ERRORS="1", PROJECT_ROOT="stale")
    bootstrap = 'export PATH="$(cygpath -u "$1"):$PATH"; exec bash "$2"' if os.name == "nt" else 'export PATH="$1:$PATH"; exec bash "$2"'
    result = execute([bash, "-c", bootstrap, "test",
                      fake_bin.as_posix(), (scripts / SUBMIT).as_posix()], env=env)
    assert result.returncode == (1 if fail_second else 0), result.stdout + result.stderr
    submissions = list((tmp_path / "scratch").rglob("jobs.tsv"))
    assert len(submissions) == 1
    rows = submissions[0].read_text().splitlines()
    assert len(rows) == (2 if fail_second else 3)
    assert rows[1].split("\t")[:2] == ["contextgraph", "65001"]
    if fail_second:
        assert "Previously submitted jobs are retained" in result.stderr
    else:
        assert rows[2].split("\t")[:2] == ["foldagent", "65002"]
    for method in ("contextgraph", "foldagent"):
        settings = (capture / f"{method}.env").read_text().splitlines()
        assert settings[:10] == ["all", "-1", "200", "65536", "compact_v1", "repaired", "1", "unset", "unset", "unset"]
        assert method in settings[10] and settings[10] != "stale"
        assert Path(settings[11]).joinpath("scripts", RUNNER).exists()
        args = (capture / f"{method}.args").read_text().splitlines()
        assert "--nodes=4" in args and "--time=24:00:00" in args and "--export=ALL" in args
        assert args[-1] == method
