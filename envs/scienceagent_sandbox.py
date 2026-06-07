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
import sys
import signal
import time
import traceback
import threading
from typing import Any


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
