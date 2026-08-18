"""Execution backends for D3-Gym task environments.

D3-Gym distributes one self-contained image per task.  This module keeps the
agent-facing interface intentionally small (``execute``, ``evaluate``,
``list_output_files``, ``close``) so it can be used by the existing code-agent
loops without teaching those loops about containers.

Supported runtimes:

* ``docker``: starts one long-lived container per trajectory.
* ``apptainer`` / ``singularity``: executes the task image for every call while
  bind-mounting the same ``pred_results`` directory.
* ``local``: copies an unpacked synthetic/task directory into the trajectory
  workdir.  This is primarily for tests and development.

Python variables do not persist between calls.  Files do, which is sufficient
for an agent to iteratively build ``solution.py`` and outputs while preserving
the task image's dependency isolation.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from envs.scienceagent_sandbox import _scan_forbidden


_STDOUT_CAP = 4096
_STDERR_CAP = 4096


def _truncate(value: str, cap: int) -> str:
    if len(value) <= cap:
        return value
    half = cap // 2
    return f"{value[:half]}\n... [{len(value) - cap} chars truncated] ...\n{value[-half:]}"


def parse_d3gym_verdict(returncode: int, stdout: str, stderr: str) -> tuple[float, str, str]:
    """Parse the verifier result emitted by a D3-Gym image.

    Official evaluators return ``(bool, detail)`` and the image entrypoint
    prints that value.  JSON and explicit PASS/FAIL output are accepted as
    defensive fallbacks.  The exit-code fallback is deliberately last and is
    surfaced in the rule name for auditability.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            value = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        detail = ""
        score_value: Any = value
        if isinstance(value, (tuple, list)) and value:
            score_value = value[0]
            if len(value) > 1:
                detail = str(value[1])
        if isinstance(score_value, bool):
            return float(score_value), "tuple" if isinstance(value, (tuple, list)) else "bool", detail
        if isinstance(score_value, (int, float)):
            return max(0.0, min(1.0, float(score_value))), "number", detail

    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            if "score" in value and isinstance(value["score"], (int, float)):
                return max(0.0, min(1.0, float(value["score"]))), "json:score", str(value.get("detail", ""))
            if "success" in value:
                return float(bool(value["success"])), "json:success", str(value.get("detail", ""))

    joined = "\n".join(lines[-5:]).lower()
    if re.search(r"\b(pass|passed|success|successful)\b", joined) and not re.search(
        r"\b(fail|failed|failure|unsuccessful)\b", joined
    ):
        return 1.0, "text:pass", joined[-500:]
    if re.search(r"\b(fail|failed|failure|unsuccessful)\b", joined):
        return 0.0, "text:fail", joined[-500:]

    tail = (stderr or stdout or "").strip()[-500:]
    return (1.0 if returncode == 0 else 0.0), "exitcode", tail


class D3GymSandbox:
    """Run agent Python and the official verifier inside one D3-Gym task."""

    def __init__(
        self,
        *,
        task_id: str,
        workdir: str,
        image: Optional[str] = None,
        runtime: str = "auto",
        task_dir: Optional[str] = None,
        per_call_timeout: float = 120.0,
        eval_timeout: float = 300.0,
        eval_script_source: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.workdir = os.path.abspath(workdir)
        self.output_dir = os.path.join(self.workdir, "pred_results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.per_call_timeout = float(per_call_timeout)
        self.eval_timeout = float(eval_timeout)
        self.call_count = 0
        self.container_name: Optional[str] = None
        self.task_dir: Optional[str] = None

        self.runtime = self._select_runtime(runtime, task_dir)
        self.image = self._resolve_image(image)
        if self.runtime == "local":
            self._prepare_local_task(task_dir, eval_script_source)
        elif self.runtime == "docker":
            self._start_docker()

    @staticmethod
    def _select_runtime(runtime: str, task_dir: Optional[str]) -> str:
        requested = (os.environ.get("D3GYM_RUNTIME") or runtime or "auto").lower()
        if requested != "auto":
            if requested not in {"docker", "apptainer", "singularity", "local"}:
                raise ValueError(f"unsupported D3-Gym runtime: {requested}")
            if requested != "local" and shutil.which(requested) is None:
                raise RuntimeError(f"D3-Gym runtime executable not found: {requested}")
            return requested
        if task_dir:
            return "local"
        for candidate in ("docker", "apptainer", "singularity"):
            if shutil.which(candidate):
                return candidate
        raise RuntimeError(
            "no D3-Gym runtime found; install docker/apptainer or set "
            "extra_info.task_dir for the local development backend"
        )

    def _resolve_image(self, image: Optional[str]) -> Optional[str]:
        if self.runtime == "local":
            return None
        image_dir = os.environ.get("D3GYM_IMAGE_DIR")
        if image_dir:
            for suffix in (".sif", ".sqsh", ""):
                candidate = os.path.join(image_dir, f"{self.task_id}{suffix}")
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)
        resolved = image or f"hananemoussa/d3-gym:{self.task_id}"
        if self.runtime in {"apptainer", "singularity"} and "://" not in resolved and not os.path.isfile(resolved):
            resolved = "docker://" + resolved
        return resolved

    def _prepare_local_task(self, task_dir: Optional[str], eval_script_source: Optional[str]) -> None:
        staged = os.path.join(self.workdir, "task")
        if task_dir:
            if not os.path.isdir(task_dir):
                raise FileNotFoundError(f"D3-Gym local task directory not found: {task_dir}")
            shutil.copytree(task_dir, staged, dirs_exist_ok=True)
        else:
            os.makedirs(staged, exist_ok=True)
        self.task_dir = staged
        local_outputs = os.path.join(staged, "pred_results")
        if os.path.lexists(local_outputs):
            if os.path.islink(local_outputs) or os.path.isfile(local_outputs):
                os.unlink(local_outputs)
            else:
                shutil.rmtree(local_outputs)
        try:
            os.symlink(self.output_dir, local_outputs, target_is_directory=True)
        except OSError:
            # Windows CI commonly lacks symlink privilege.  Use the task-local
            # directory and point output_dir at it instead.
            os.makedirs(local_outputs, exist_ok=True)
            self.output_dir = local_outputs
        if eval_script_source and not os.path.isfile(os.path.join(staged, "eval_script.py")):
            Path(staged, "eval_script.py").write_text(eval_script_source, encoding="utf-8")

    def _start_docker(self) -> None:
        assert self.image
        self.container_name = f"d3gym_{self.task_id}_{uuid.uuid4().hex[:10]}"
        mount = f"{self.output_dir}:/task/pred_results"
        proc = subprocess.run(
            [
                "docker", "run", "-d", "--rm", "--name", self.container_name,
                "--network", "none", "--entrypoint", "sh", "-v", mount,
                self.image, "-c", "while true; do sleep 3600; done",
            ],
            capture_output=True,
            text=True,
            timeout=max(self.per_call_timeout, 300.0),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"could not start D3-Gym image {self.image}: {(proc.stderr or proc.stdout)[-1000:]}")

    def _base_exec(self) -> list[str]:
        if self.runtime == "docker":
            assert self.container_name
            return ["docker", "exec", "-w", "/task", self.container_name]
        if self.runtime in {"apptainer", "singularity"}:
            assert self.image
            return [
                self.runtime, "exec", "--cleanenv", "--bind",
                f"{self.output_dir}:/task/pred_results", "--pwd", "/task", self.image,
            ]
        return []

    def _run(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        started = time.time()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                argv, 124, exc.stdout or "", (exc.stderr or "") + f"\n[D3-Gym] timed out after {timeout}s"
            )
        proc.elapsed = time.time() - started  # type: ignore[attr-defined]
        return proc

    def execute(self, code: str) -> dict[str, Any]:
        self.call_count += 1
        blocked = _scan_forbidden(code)
        if blocked is not None:
            return {
                "stdout": "",
                "stderr": f"[D3-Gym sandbox] blocked unsafe operation: {blocked}",
                "success": False,
                "elapsed": 0.0,
                "call_count": self.call_count,
            }
        if self.runtime == "local":
            assert self.task_dir
            argv = [sys.executable, "-c", code]
            started = time.time()
            try:
                proc = subprocess.run(
                    argv, cwd=self.task_dir, capture_output=True, text=True,
                    timeout=self.per_call_timeout,
                )
                elapsed = time.time() - started
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(argv, 124, exc.stdout or "", exc.stderr or "")
                elapsed = time.time() - started
        else:
            proc = self._run(self._base_exec() + ["python", "-c", code], self.per_call_timeout)
            elapsed = getattr(proc, "elapsed", 0.0)
        return {
            "stdout": _truncate(proc.stdout or "", _STDOUT_CAP),
            "stderr": _truncate(proc.stderr or "", _STDERR_CAP),
            "success": proc.returncode == 0,
            "elapsed": elapsed,
            "call_count": self.call_count,
        }

    def evaluate(self) -> dict[str, Any]:
        code = "import eval_script; print(eval_script.eval())"
        if self.runtime == "local":
            assert self.task_dir
            argv = [sys.executable, "-c", code]
            try:
                proc = subprocess.run(
                    argv, cwd=self.task_dir, capture_output=True, text=True,
                    timeout=self.eval_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(argv, 124, exc.stdout or "", exc.stderr or "")
        else:
            proc = self._run(self._base_exec() + ["python", "-c", code], self.eval_timeout)
        score, rule, detail = parse_d3gym_verdict(proc.returncode, proc.stdout or "", proc.stderr or "")
        tail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        return {
            "score": score,
            "rule": rule,
            "detail": detail or tail,
            "returncode": proc.returncode,
            "stdout": _truncate(proc.stdout or "", _STDOUT_CAP),
            "stderr": _truncate(proc.stderr or "", _STDERR_CAP),
        }

    def list_output_files(self) -> list[str]:
        root = Path(self.output_dir)
        if not root.is_dir():
            return []
        return sorted(str(path.relative_to(root)).replace(os.sep, "/") for path in root.rglob("*") if path.is_file())

    def list_input_files(self) -> list[str]:
        """Return paths exactly as the agent can open them from ``/task``."""
        if self.runtime == "local":
            assert self.task_dir
            root = Path(self.task_dir, "datasets")
            if not root.is_dir():
                return []
            return sorted(
                "datasets/" + str(path.relative_to(root)).replace(os.sep, "/")
                for path in root.rglob("*")
                if path.is_file()
            )
        code = (
            "import json,os; print(json.dumps(sorted(" 
            "os.path.relpath(os.path.join(r,f), '/task').replace(os.sep,'/') "
            "for r,_,fs in os.walk('/task/datasets') for f in fs)))"
        )
        proc = self._run(self._base_exec() + ["python", "-c", code], min(self.per_call_timeout, 120.0))
        if proc.returncode != 0:
            return []
        try:
            value = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except (IndexError, TypeError, ValueError):
            return []
        return [str(path) for path in value if isinstance(path, str)]

    def close(self) -> None:
        if self.runtime == "docker" and self.container_name:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.container_name = None
