"""Real success-rate scorer for ScienceAgentBench tasks.

Replaces the Phase-D2 file-existence placeholder in
`ScienceAgentEnv.get_reward`. Each task ships a per-task evaluation script
(`eval_programs/<eval_script_name>`) that inspects the file the agent wrote
to `pred_results/<output_fname>` and decides whether the scientific result
is correct.

────────────────────────────────────────────────────────────────────────
CONTRACT (confirmed against upstream eval_programs/clintox_nn_eval.py):

  Every SAB eval script ends with:
      def eval():
          pred = pd.read_csv('pred_results/<output_fname>')        # agent output
          gold = pd.read_csv('benchmark/eval_programs/gold_results/...')  # gold ref
          ...
          return int(success), str(detail)
      if __name__ == "__main__":
          print(eval())

  So the script:
    - is run with cwd = a dir that has BOTH `pred_results/` (the agent's
      output) AND `benchmark/` (pointing at the benchmark root, where the
      gold reference lives under eval_programs/gold_results/). We satisfy
      this by running in the task workdir with a `benchmark` symlink ->
      benchmark_dir.
    - emits its verdict as a printed Python tuple repr `(0|1|float, "...")`
      on stdout. CRITICAL: the script exits 0 even when the task FAILS
      (it just prints `(0, ...)`), so exit code is NOT a success signal —
      we MUST read the first tuple element.

  `_parse_verdict` reads the score in priority order:
    1. ast.literal_eval of a stdout line yielding a tuple/list -> elem[0].  <- SAB
    2. ast.literal_eval yielding a bare number -> that number.
    3. JSON object with "success"/"score".                          <- defensive
    4. process exit code (0 -> 1.0).                                <- last resort
  Which rule fired is logged in `detail` so a run can be audited.
────────────────────────────────────────────────────────────────────────

This runs ENV-SIDE at reward time (not agent-emitted code), so unlike the
sandbox it is allowed to use subprocess. It is deliberately isolated in a
child process: eval scripts import heavy scientific libs and may chdir /
mutate global state we do not want leaking into the Ray worker.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from typing import Optional


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def locate_eval_script(benchmark_dir: str, eval_script_name: str) -> Optional[str]:
    """Resolve the absolute path of a task's eval script.

    Upstream ScienceAgentBench keeps these under `eval_programs/`; we also
    probe a couple of sibling layouts so a slightly different unpack of the
    (password-protected) benchmark zip still works.
    """
    if not benchmark_dir or not eval_script_name:
        return None
    candidates = [
        os.path.join(benchmark_dir, "eval_programs", eval_script_name),
        os.path.join(benchmark_dir, "eval", eval_script_name),
        os.path.join(benchmark_dir, eval_script_name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _coerce_score(value) -> Optional[float]:
    """Turn the first element of an eval verdict into a [0,1] float.

    Handles int(0/1), float partial credit, and bool. Returns None if the
    value isn't numeric (so the caller can fall through to the next rule).
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_verdict(returncode: int, stdout: str, stderr: str) -> tuple[float, str]:
    """Extract a [0,1] score from a finished eval subprocess.

    Returns (score, which_rule_fired). See module docstring for the
    contract. Rules are in priority order; the FIRST that matches wins so
    the convention used is unambiguous and auditable.
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]

    # Rule 1/2: SAB convention — `print(eval())` where eval() returns
    # `(int_score, detail_str)`. Parse the last literal-eval-able line; a
    # tuple/list yields elem[0], a bare number yields itself.
    for line in reversed(lines):
        try:
            obj = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(obj, (tuple, list)) and obj:
            s = _coerce_score(obj[0])
            if s is not None:
                return s, "tuple"
        s = _coerce_score(obj)
        if s is not None:
            return s, "number"

    # Rule 3: defensive — a JSON object with success/score (not seen in the
    # confirmed sample, kept in case some eval scripts differ).
    for line in reversed(lines):
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            o = json.loads(line)
        except (ValueError, TypeError):
            continue
        if "score" in o and _coerce_score(o["score"]) is not None:
            return _coerce_score(o["score"]), "json:score"
        if "success" in o:
            return (1.0 if o["success"] else 0.0), "json:success"

    # Rule 4: exit code — LAST resort. SAB scripts exit 0 even on task
    # FAILURE, so reaching here means we could not read the printed verdict;
    # a run that hits this often is a red flag to audit, not a pass.
    return (1.0 if returncode == 0 else 0.0), "exitcode"


def score_task(
    workdir: str,
    benchmark_dir: str,
    eval_script_name: str,
    output_fname: Optional[str],
    *,
    timeout: float = 180.0,
    python_exe: Optional[str] = None,
) -> dict:
    """Score one finished trajectory.

    Args:
        workdir: the trajectory workdir; the agent wrote
            `pred_results/<output_fname>` here.
        benchmark_dir: root of the unpacked SAB benchmark (has
            `eval_programs/`, `datasets/`, `gold_programs/`).
        eval_script_name: file name of this task's eval script.
        output_fname: the relative path the task asked the agent to produce
            (used for the valid-execution check).
        timeout: per-task wall-clock cap for the eval subprocess.
        python_exe: interpreter to run the eval with (defaults to the
            current one; pass a task conda env's python if eval deps differ).

    Returns dict:
        score (float 0..1), valid_execution (bool), rule (str),
        detail (str), eval_found (bool).
    """
    # The evaluator changes cwd to the task workdir. Resolve caller-provided
    # relative paths first so eval_path cannot accidentally become relative
    # to that scratch directory.
    workdir = os.path.abspath(workdir)
    benchmark_dir = os.path.abspath(benchmark_dir)

    # Valid-execution = the expected artifact is actually on disk. This is
    # the same signal the old placeholder used, kept here as a separate axis
    # (VER) alongside the real success score (SR).
    out_path = (
        os.path.join(workdir, "pred_results", os.path.basename(output_fname))
        if output_fname else None
    )
    valid_execution = bool(out_path and os.path.isfile(out_path))

    eval_path = locate_eval_script(benchmark_dir, eval_script_name)
    if eval_path is None:
        return {
            "score": 0.0,
            "valid_execution": valid_execution,
            "rule": "no-eval-script",
            "detail": f"eval script {eval_script_name!r} not found under {benchmark_dir!r}",
            "eval_found": False,
        }

    if not valid_execution:
        # No output -> the eval script would just error on a missing file.
        # Short-circuit so we do not blame the eval contract for a no-output
        # trajectory.
        return {
            "score": 0.0,
            "valid_execution": False,
            "rule": "no-output",
            "detail": f"expected output {output_fname!r} absent in pred_results/",
            "eval_found": True,
        }

    # The eval script reads gold via the RELATIVE path
    # `benchmark/eval_programs/gold_results/...`, so its cwd must expose a
    # `benchmark` dir pointing at the benchmark root (alongside the agent's
    # `pred_results/`, already in workdir). Stage that symlink.
    bench_link = os.path.join(workdir, "benchmark")
    if not os.path.exists(bench_link):
        try:
            os.symlink(benchmark_dir, bench_link, target_is_directory=True)
        except OSError as e:
            # Windows without developer mode forbids symlinks (WinError 1314);
            # a directory junction needs no privilege. Production is Linux, so
            # this fallback only matters for local dev testing.
            linked = False
            if os.name == "nt":
                try:
                    subprocess.run(["cmd", "/c", "mklink", "/J", bench_link,
                                    benchmark_dir], capture_output=True, check=True)
                    linked = os.path.exists(bench_link)
                except Exception:  # noqa: BLE001
                    linked = False
            if not linked:
                return {
                    "score": 0.0,
                    "valid_execution": valid_execution,
                    "rule": "benchmark-link-failed",
                    "detail": f"could not stage benchmark symlink: {e}",
                    "eval_found": True,
                }

    env = dict(os.environ)
    # Upstream keeps shared evaluator helpers (notably
    # gpt4_visual_judge.py) at the repository root, outside benchmark/.
    # Include our project root explicitly because Ray workers execute from
    # per-task scratch directories and cannot otherwise resolve that module.
    support_dir = os.environ.get("SAB_EVAL_SUPPORT_DIR", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            os.path.dirname(eval_path),
            benchmark_dir,
            support_dir,
            _PROJECT_ROOT,
            env.get("PYTHONPATH", ""),
        )
        if p
    )
    env["SAB_BENCHMARK_DIR"] = benchmark_dir

    try:
        # Run the eval by absolute path with cwd=workdir so its relative
        # `pred_results/` and `benchmark/` reads both resolve.
        proc = subprocess.run(
            [python_exe or sys.executable, eval_path],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "score": 0.0,
            "valid_execution": valid_execution,
            "rule": "eval-timeout",
            "detail": f"eval script exceeded {timeout}s",
            "eval_found": True,
        }
    except Exception as e:  # noqa: BLE001 - never let scoring crash the rollout
        return {
            "score": 0.0,
            "valid_execution": valid_execution,
            "rule": "eval-crashed",
            "detail": f"{type(e).__name__}: {e}",
            "eval_found": True,
        }

    score, rule = _parse_verdict(proc.returncode, proc.stdout, proc.stderr)
    score = max(0.0, min(1.0, score))
    tail = (proc.stderr or proc.stdout or "").strip()[-300:]
    return {
        "score": score,
        "valid_execution": valid_execution,
        "rule": rule,
        "detail": f"rc={proc.returncode} rule={rule} :: {tail}",
        "eval_found": True,
    }
