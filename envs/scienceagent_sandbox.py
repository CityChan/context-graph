"""Stateful Python sandbox for ScienceAgentBench-style code agents.

Implementation:
    A single `exec()` namespace per trajectory, run in-process. stdout/stderr
    are captured via direct sys.stdout swap with try/finally restore.

    Per-call timeout: signal.SIGALRM when invoked from the main thread
    (Vista production path); silently disabled when invoked from a
    background thread (Windows / pytest path) because signal.SIGALRM is
    main-thread-only. The agent is responsible for keeping individual
    python_exec calls short (the system prompt instructs this).

This is NOT a security sandbox. It runs untrusted LLM-emitted code in the
same Python process. For Vista eval this is acceptable because:
  - The SLURM job is process-isolated already.
  - We do not give agents shell access (no os.system, no subprocess; just
    Python libraries).
  - We do not give agents external network access via this tool.

For an adversarial deployment, swap to the OpenHands docker sandbox. The
interface (`execute(code) -> dict`) is unchanged.
"""

from __future__ import annotations

import io
import os
import re
import sys
import signal
import time
import traceback
import threading
from typing import Any


# ── Hard guard: shell / network / package-install are disabled ──
#
# The system prompt already tells the agent not to shell out or pip-install,
# but weak (8B) models ignore that — the first smoke saw a task run
# `pip install DeepPurpose` inside the sandbox (17s wasted, non-deterministic,
# pollutes the run). We enforce the rule here: code matching any pattern below
# is rejected BEFORE exec() with a corrective stderr the agent can route
# around. Every scientific package the 102 tasks need must be pre-installed in
# the conda env (see requirements_sab_missing.txt for the ones to add).
#
# Static string scan (not AST) is deliberate: it runs before exec with zero
# runtime risk, is deterministic/testable, and a rare false positive just
# costs one rejected call with a clear message — acceptable for a benchmark.
_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bpip\s+install\b"), "pip install"),
    (re.compile(r"\bpip3\s+install\b"), "pip3 install"),
    (re.compile(r"-m\s+pip\b"), "python -m pip"),
    (re.compile(r"!\s*pip\b"), "!pip magic"),
    (re.compile(r"%\s*pip\b"), "%pip magic"),
    (re.compile(r"\bconda\s+install\b"), "conda install"),
    (re.compile(r"\bsubprocess\b"), "subprocess"),
    (re.compile(r"\bos\.system\s*\("), "os.system(...)"),
    (re.compile(r"\bos\.popen\s*\("), "os.popen(...)"),
    (re.compile(r"\bos\.spawn\w*\s*\("), "os.spawn*(...)"),
    (re.compile(r"\bpty\.spawn\s*\("), "pty.spawn(...)"),
    (re.compile(r"\bsys\.executable\b"), "sys.executable"),
    (re.compile(r"\bget_ipython\s*\("), "get_ipython()"),
]

_BLOCK_MSG = (
    "[Sandbox] Blocked: detected `{what}`. Shell access, subprocess spawning, "
    "network, and package installation are DISABLED in this sandbox. Every "
    "scientific package the tasks need (numpy, pandas, scikit-learn, scipy, "
    "torch, scanpy, anndata, rdkit, deepchem, DeepPurpose, matplotlib, seaborn, "
    "xgboost, statsmodels, geopandas, rasterio, ...) is ALREADY installed — "
    "just `import` what you need directly. Do NOT install anything or call the "
    "shell."
)


def _scan_forbidden(code: str) -> str | None:
    """Return the human label of the first forbidden pattern in `code`, or None."""
    for pat, what in _FORBIDDEN_PATTERNS:
        if pat.search(code):
            return what
    return None


class CodeSandbox:
    """A persistent Python execution namespace scoped to one trajectory.

    Usage:
        sb = CodeSandbox(workdir='/tmp/task_69/', per_call_timeout=60)
        out = sb.execute("import pandas as pd; df = pd.read_csv('data.csv'); print(df.shape)")
        # out: {'stdout': '(20000, 33538)\n', 'stderr': '', 'success': True, 'elapsed': 0.3}
        sb.close()
    """

    STDOUT_CAP = 2048
    STDERR_CAP = 2048

    def __init__(
        self,
        workdir: str,
        per_call_timeout: float = 60.0,
    ):
        if not os.path.isdir(workdir):
            os.makedirs(workdir, exist_ok=True)
        self.workdir = os.path.abspath(workdir)
        # Ensure pred_results/ exists (tasks write here)
        os.makedirs(os.path.join(self.workdir, "pred_results"), exist_ok=True)

        self.per_call_timeout = per_call_timeout
        # Persistent namespace for exec()
        self.namespace: dict = {
            "__name__": "__sandbox__",
            "__file__": os.path.join(self.workdir, "<sandbox>"),
        }
        self.call_count = 0
        # We chdir into workdir on every execute() (in case agent code touched cwd)
        self._original_cwd = os.getcwd()

    def execute(self, code: str) -> dict[str, Any]:
        """Execute `code`. Returns dict with stdout/stderr/success/elapsed.

        stdout/stderr are captured via sys.stdout/stderr swap inside a
        try/finally so the originals are always restored. Timeout uses
        signal.SIGALRM when the call is on the main thread; otherwise
        timeout is silently disabled (best-effort).
        """
        self.call_count += 1

        # Hard guard: reject shell / pip / subprocess before running anything.
        blocked = _scan_forbidden(code)
        if blocked is not None:
            return {
                "stdout": "",
                "stderr": _BLOCK_MSG.format(what=blocked),
                "success": False,
                "elapsed": 0.0,
                "call_count": self.call_count,
            }

        buf_out = io.StringIO()
        buf_err = io.StringIO()

        start = time.time()
        timed_out = False
        ok = True
        tb_text: str = ""

        prev_cwd = os.getcwd()
        try:
            os.chdir(self.workdir)
        except OSError:
            pass

        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf_out, buf_err

        # signal.SIGALRM is main-thread-only on POSIX. asyncio's default
        # executor puts our coroutine on the main thread, so this works in
        # production. On Windows or from a thread, fall back to no-timeout.
        can_use_alarm = (
            threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
        )

        def _alarm_handler(signum, frame):
            raise TimeoutError(f"sandbox call exceeded {self.per_call_timeout}s")

        prev_alarm = None
        if can_use_alarm:
            prev_alarm = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, self.per_call_timeout)

        try:
            exec(compile(code, "<sandbox>", "exec"), self.namespace)
        except TimeoutError as e:
            timed_out = True
            ok = False
            tb_text = str(e)
        except BaseException:
            ok = False
            tb_text = traceback.format_exc()
        finally:
            if can_use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                if prev_alarm is not None:
                    signal.signal(signal.SIGALRM, prev_alarm)
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            try:
                os.chdir(prev_cwd)
            except OSError:
                pass

        elapsed = time.time() - start
        stdout = buf_out.getvalue()
        stderr = buf_err.getvalue()
        if tb_text:
            stderr = (stderr + "\n" + tb_text).strip()
        if timed_out:
            stderr = (stderr + f"\n[Sandbox] Execution timed out after {self.per_call_timeout}s.").strip()

        return {
            "stdout": _truncate(stdout, self.STDOUT_CAP),
            "stderr": _truncate(stderr, self.STDERR_CAP),
            "success": ok and not timed_out,
            "elapsed": elapsed,
            "call_count": self.call_count,
        }

    def close(self) -> None:
        """Drop references so GC can reclaim the namespace."""
        self.namespace.clear()

    def list_output_files(self) -> list[str]:
        """Files the agent wrote to pred_results/ (used for reward scoring)."""
        out_dir = os.path.join(self.workdir, "pred_results")
        if not os.path.isdir(out_dir):
            return []
        return sorted(os.listdir(out_dir))


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    head = s[: n // 2]
    tail = s[-n // 2 :]
    return f"{head}\n... [{len(s) - n} chars truncated] ...\n{tail}"


# ── Import pre-warming (event-loop-stall fix) ──
#
# The sandbox runs `exec()` SYNCHRONOUSLY on the agent loop's event-loop thread.
# When an agent does a heavy native import (e.g. `import deepchem`, which drags
# in tensorflow + probes jax/torch_geometric), that import blocks the event
# loop for tens of seconds — starving every OTHER trajectory on the same
# AgentLoopWorker, and neither the outer asyncio.wait_for(120s) (loop is
# blocked, can't fire) nor the sandbox SIGALRM(60s) (deferred during C-level
# imports) can interrupt it. With 102 tasks each re-importing the same heavy
# libs, the serialized import time reads as a multi-minute hang.
#
# Fix: import the heavy packages ONCE per worker process at agent-loop
# init_class time. `sys.modules` is process-global, so every later
# `import X` inside a sandbox is then an instant dict hit that never blocks the
# loop. The one-time cost is paid visibly at worker startup, not mid-eval.
#
# This does NOT fix slow non-import compute blocking the loop — the durable fix
# for that is a subprocess-based sandbox (design doc §5), where each trajectory
# is a killable child process. Pre-warming just removes the dominant stall
# (repeated heavy imports) that the first 4-node smoke hit.

_PREWARM_DONE = False

# Ordered cheap→heavy. matplotlib is forced to the headless Agg backend first
# (tasks save figures, never display). Each import is best-effort: a missing or
# broken package is skipped, never fatal to worker startup.
_PREWARM_PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn",
    "statsmodels", "xgboost",
    "matplotlib", "matplotlib.pyplot", "seaborn",
    "torch",
    "rdkit", "rdkit.Chem",
    "scanpy", "anndata",
    "deepchem", "DeepPurpose",
    "geopandas", "rasterio",
]


def prewarm_heavy_imports(verbose: bool = True) -> None:
    """Import heavy scientific packages once per process so per-trajectory
    `import X` in the sandbox is an instant sys.modules cache hit and never
    blocks the event loop. Idempotent; safe to call from every agent-loop
    init_class."""
    global _PREWARM_DONE
    if _PREWARM_DONE:
        return
    _PREWARM_DONE = True

    import importlib
    import time

    # CRITICAL: the AgentLoopWorker shares its single GH200 with vLLM. deepchem
    # drags in tensorflow, which by default pre-allocates ALL GPU memory and
    # would OOM the vLLM engine. Force on-demand growth and silence the import
    # log spam BEFORE any of these libs is imported. setdefault so an explicit
    # sbatch-level override still wins.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # jax, if pulled

    # Headless figure backend before pyplot is imported anywhere.
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

    t0 = time.time()
    warmed, skipped = 0, 0
    for name in _PREWARM_PACKAGES:
        t = time.time()
        try:
            importlib.import_module(name)
            warmed += 1
            if verbose and time.time() - t > 1.0:
                print(f"[SAB prewarm] {name} in {time.time() - t:.1f}s")
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"[SAB prewarm] skip {name}: {type(e).__name__}: {e}")
    if verbose:
        print(f"[SAB prewarm] done: {warmed} warmed, {skipped} skipped, "
              f"{time.time() - t0:.1f}s total")
