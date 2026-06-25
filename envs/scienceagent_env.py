"""Env wrapper for ScienceAgentBench-style tasks.

Mirrors envs/local_search.py's LocalSearch interface so it plugs into the
existing agent loops (react_agent, fold_agent, graph_agent_isolated)
without changes to the loop code — the loop just dispatches on
config.plugin.workflow which we set to `code` / `code_branch` / `code_graph`.

Selection: ability string `ScienceAgentBench` (added to
agents/utils.py:select_env in a follow-up patch).

Per-task lifecycle:
  __init__(config, tokenizer, ability)
  await init_env(item)             # reads task spec, sets up workdir, spawns sandbox
  await run_action(response)       # dispatches LLM tool call to sandbox
  await get_reward(item, msgs, ctx) # scores by checking pred_results/
"""

from __future__ import annotations

import collections
import copy
import os
import re
import shutil
import tempfile
from typing import Any

import numpy as np

# Re-use the existing XML-tag tool-call extractor.
# (defined in envs/local_search.py — small enough to import without circular issues)
from envs.local_search import extract_fn_call

from envs.scienceagent_sandbox import CodeSandbox


class ScienceAgentEnv:
    """One env instance per trajectory. Each instance has its own sandbox."""

    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability

        # Stats follow the LocalSearch convention so the existing reward
        # manager / wandb aggregation works out of the box.
        self.stats = collections.Counter()
        for k in ('action', 'finish', 'finish_rejected', 'python_exec',
                  'is_finish', 'env_error'):
            self.stats[k] = 0

        self.env_fail = False
        self.is_finish = False
        self.predicted_answer = None  # "answer" is the path(s) of output files

        # Filled in by init_env
        self.task_id: str | None = None
        self.instruction: str | None = None
        self.input_files: list[str] = []        # source paths (read-only)
        self.expected_output: str | None = None  # relative path the task asks for
        self.gold_eval_script: str | None = None  # back-compat alias of eval_script_name
        self.eval_script_name: str | None = None  # per-task eval script (real scoring)
        self.benchmark_dir: str | None = None     # unpacked benchmark root
        self.workdir: str | None = None
        self.sandbox: CodeSandbox | None = None
        self.instance_info: dict = {}
        self.input_manifest: list[str] = []     # rel paths actually on disk in workdir
        self.expected_output_basename: str | None = None
        self.eval_contract: str | None = None
        # Single-call time limit (overridable per-task via extra_info)
        self.per_call_timeout = getattr(getattr(config, 'plugin', object()),
                                        "sandbox_timeout", 60.0)

    async def init_env(self, item):
        extra = item.non_tensor_batch.get('extra_info', None)
        # verl may wrap as a 0-d numpy array
        if hasattr(extra, 'ndim'):
            extra = extra.item() if extra.ndim == 0 else extra[0]

        if extra is None:
            self.env_fail = True
            self.instance_info = {'problem_statement': '(missing task spec)'}
            return

        self.task_id = str(extra.get('task_id', 'unknown'))
        self.instruction = extra.get('instruction') or extra.get('query', '')
        self.input_files = list(extra.get('input_files', []))
        # Relative paths (e.g. "dkpes/dkpes_train.csv") the task references; the
        # file MUST be placed at workdir/<rel>, not flattened to its basename,
        # or the agent's open("dkpes/dkpes_train.csv") fails. Parallel to
        # input_files; falls back to basename if absent (back-compat).
        self.input_rel_paths = list(extra.get('input_rel_paths', []) or [])
        self.expected_output = extra.get('expected_output', None)
        # NOTE: the loader emits `eval_script_name` (not `gold_eval_script`);
        # the latter was never set, so real-eval scoring silently no-op'd.
        # Read the real key, keep the old one as a back-compat fallback.
        self.eval_script_name = (
            extra.get('eval_script_name') or extra.get('gold_eval_script') or None
        )
        self.gold_eval_script = self.eval_script_name  # back-compat alias
        # Root of the unpacked benchmark (eval_programs/, datasets/, ...).
        # Needed at reward time to locate the per-task eval script.
        self.benchmark_dir = extra.get('benchmark_dir', None)

        # The instance_info dict is consumed by create_chat to build the user prompt.
        self.instance_info = copy.deepcopy(extra)
        self.instance_info['problem_statement'] = self.instruction

        # Set up a unique workdir for this trajectory.
        # If extra['workdir'] is given, USE it (production: caller sets a
        # per-trajectory dir under SCRATCH). Otherwise mkdtemp (smoke/local).
        self.workdir = extra.get('workdir') or tempfile.mkdtemp(
            prefix=f"sab_{self.task_id}_"
        )
        os.makedirs(self.workdir, exist_ok=True)

        # Copy (or symlink, for large files) input files into workdir, PRESERVING
        # the relative directory structure the task instruction references.
        missing = 0
        placed_rel: list[str] = []  # rel paths actually present in workdir
        for i, src in enumerate(self.input_files):
            rel = (self.input_rel_paths[i]
                   if i < len(self.input_rel_paths) and self.input_rel_paths[i]
                   else os.path.basename(src))
            if not os.path.exists(src):
                missing += 1
                print(f"[SAB env] WARNING task {self.task_id}: input file missing "
                      f"on disk, agent will not see it: {src}")
                continue
            dst = os.path.join(self.workdir, rel)
            placed_rel.append(rel)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst) or self.workdir, exist_ok=True)
            try:
                # Symlink for speed when possible; fall back to copy on Windows etc.
                os.symlink(src, dst)
            except (OSError, AttributeError):
                shutil.copy2(src, dst)
        if missing:
            print(f"[SAB env] task {self.task_id}: {missing}/{len(self.input_files)} "
                  f"declared input files were absent on disk")

        # Build a ground-truth file manifest from what is ACTUALLY on disk in
        # the workdir, so the prompt can tell the agent the exact relative paths
        # to open() instead of letting it guess (the 8B smoke showed the agent
        # burning turns probing `atlantic_profiles.nc` -> `ocean_profiles/...`
        # -> ... because nothing told it the real layout). Cap the list so a
        # task with hundreds of image tiles can't blow the prompt budget.
        self.input_manifest = placed_rel
        self.expected_output_basename = (
            os.path.basename(self.expected_output) if self.expected_output else None
        )
        self.eval_contract = self._maybe_build_eval_contract()

        self.sandbox = CodeSandbox(self.workdir, per_call_timeout=self.per_call_timeout)

    async def run_action(self, response: str) -> dict:
        self.stats['action'] += 1
        if self.env_fail or self.sandbox is None:
            return {'observation': '[Error] Env is in a failed state.'}

        fn_call = extract_fn_call(response)
        if not fn_call:
            return {'observation': 'No function call was detected in the model response.'}

        observation = ''
        for fn in fn_call:
            name = fn.get('function', '')
            args = fn.get('arguments', {}) or {}

            if name == 'python_exec':
                self.stats['python_exec'] += 1
                code = args.get('code', '')
                if not code:
                    observation += '[Error] The "python_exec" function requires a "code" argument.\n'
                    continue
                result = self.sandbox.execute(code)
                observation += _format_exec_result(result)
                observation += self._format_output_status()

            elif name == 'finish':
                msg = args.get('message', '(no message)')
                produced = self.sandbox.list_output_files()
                expected_name = self.expected_output_basename
                if expected_name and expected_name not in produced:
                    self.stats['finish_rejected'] += 1
                    return {
                        'observation': (
                            "[finish rejected] The required output file is not "
                            f"present: pred_results/{expected_name}. "
                            "Use python_exec to create, copy, or rename the "
                            "required file before calling finish.\n"
                            f"{self._format_output_status().strip()}"
                        )
                    }
                self.stats['finish'] += 1
                self.stats['is_finish'] = 1
                self.is_finish = True
                self.predicted_answer = (
                    msg,
                    produced,
                    self.expected_output,
                )
                return {'action': 'finish'}

            else:
                # Other tools (branch / merge / add_edge / etc.) are handled
                # by the agent loop itself, not by the env. If we reach here
                # with an unknown name it's a prompt error or an unsupported
                # tool — surface clearly.
                observation += f'[Error] The function "{name}" is not handled by ScienceAgentEnv.\n'

        return {'observation': observation.strip()}

    async def get_reward(self, item, messages, context):
        """Score the finished trajectory.

        Two modes, selected by the `SAB_REAL_EVAL` env var:

          SAB_REAL_EVAL=1 -> run the task's per-task eval script
            (`eval_programs/<eval_script_name>`) against the agent's output
            and use its [0,1] success score. Requires `benchmark_dir` to be
            plumbed through extra_info (loader passes it). This is the
            paper-grade metric.

          unset / 0 (default) -> Phase-D2 file-existence placeholder: 1.0 if
            the expected_output file name appears in pred_results/, else 0.0.
            Kept as the harness-validation path; its numbers are NOT real
            scientific correctness.
        """
        if self.env_fail or self.sandbox is None:
            return "", 0.0, {}

        produced = self.sandbox.list_output_files()
        use_real = os.environ.get("SAB_REAL_EVAL", "").lower() in ("1", "true", "yes")

        if use_real and self.eval_script_name and self.benchmark_dir:
            from envs.scienceagent_eval import score_task
            res = score_task(
                workdir=self.workdir,
                benchmark_dir=self.benchmark_dir,
                eval_script_name=self.eval_script_name,
                output_fname=self.expected_output,
                timeout=float(getattr(getattr(self.config, 'plugin', object()),
                                      'eval_timeout', 180.0)),
            )
            # Audit line: make the scoring path visible per task so a全0 run
            # can be diagnosed as real-failure vs eval-infra-breakage (rule
            # tuple/number = eval ran; eval-crashed/no-eval-script/no-output
            # = artifact). Without this the reward is an opaque 0.0.
            print(f"[SAB eval] task={self.task_id} score={res['score']:.3f} "
                  f"rule={res['rule']} valid_exec={int(res['valid_execution'])} "
                  f"produced={len(produced)} :: {res['detail'][:200]}")
            metrics = {
                "produced_files": len(produced),
                "valid_execution": int(res["valid_execution"]),
                "eval_rule": res["rule"],
            }
            return res["detail"], float(res["score"]), metrics

        if use_real:
            # Asked for real eval but a prerequisite is missing — make the
            # gap loud rather than silently degrading to placeholder scores
            # (the exact "fake win" failure mode we must avoid).
            why = ("no eval_script_name" if not self.eval_script_name
                   else "no benchmark_dir")
            print(f"[SAB env] task {self.task_id}: SAB_REAL_EVAL set but {why}; "
                  f"falling back to file-existence placeholder")

        # ── file-existence placeholder ──
        if not produced:
            return "no output files", 0.0, {"produced_files": 0}

        expected = self.expected_output
        if expected is None:
            # No expected_output specified: any file = 1.0 (smoke mode)
            return "ok (smoke: any-file-present)", 1.0, {
                "produced_files": len(produced)
            }

        expected_name = os.path.basename(expected)
        if expected_name in produced:
            return f"produced {expected_name}", 1.0, {"produced_files": len(produced)}
        return f"expected {expected_name}, got {produced}", 0.0, {
            "produced_files": len(produced),
            "expected_missing": expected_name,
        }

    async def update_dataproto(self, out, item, messages, score, reward_dict, tag='main', metrics=None):
        """Match LocalSearch.update_dataproto interface."""
        final_score = score[1]
        out.meta_info["inference_metrics"] = metrics
        out.meta_info["generation_kwargs"] = item.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(item.non_tensor_batch)
        out.non_tensor_batch["num_of_turns"] = np.array([len(messages)], dtype=object)
        out.non_tensor_batch["turn_clipped"] = np.array([False], dtype=object)
        out.non_tensor_batch["tag"] = np.array([tag], dtype=object)
        out.non_tensor_batch["is_summary"] = np.array([int("summary" in tag)], dtype=object)
        out.non_tensor_batch["traj_cnt"] = np.array([1], dtype=object)
        stats = dict(self.stats)
        stats['score'] = final_score
        extra_data = {
            "score": score,
            "call_fail": self.env_fail,
            "action_fail": 0,
            "answer_reached": True,
            "stats": stats,
        }
        out.non_tensor_batch['extra_data'] = np.array([extra_data], dtype=object)
        return out

    def close(self) -> None:
        if self.sandbox is not None:
            self.sandbox.close()

    def _format_output_status(self) -> str:
        """Compact status appended after code execution / rejected finish.

        This gives the model immediate, machine-checkable feedback about the
        only artifact condition that gates `finish`: whether the exact expected
        output file exists under pred_results/.
        """
        if self.sandbox is None:
            return "\n[output_status] sandbox unavailable\n"
        produced = self.sandbox.list_output_files()
        expected_name = self.expected_output_basename
        if expected_name:
            present = "yes" if expected_name in produced else "no"
            return (
                f"\n[output_status] expected=pred_results/{expected_name} "
                f"present={present} produced={produced}\n"
            )
        return f"\n[output_status] no expected output declared produced={produced}\n"

    def _maybe_build_eval_contract(self) -> str | None:
        """Optionally expose output-schema hints extracted from the eval script.

        This is intentionally gated because eval scripts are benchmark-side
        artifacts. Use it for diagnosis / ablations after no-output is fixed;
        leave it off for paper-faithful runs.
        """
        expose = os.environ.get("SAB_EXPOSE_EVAL_CONTRACT", "").lower()
        expose = expose in ("1", "true", "yes")
        cfg_expose = bool(getattr(getattr(self.config, 'plugin', object()),
                                  "expose_eval_contract", False))
        if not (expose or cfg_expose):
            return None
        if not (self.benchmark_dir and self.eval_script_name):
            return None
        try:
            from envs.scienceagent_eval import locate_eval_script
            eval_path = locate_eval_script(self.benchmark_dir, self.eval_script_name)
            if not eval_path:
                return None
            with open(eval_path, encoding="utf-8", errors="replace") as f:
                src = f.read()
        except Exception as e:  # noqa: BLE001 - prompt hint must never break env init
            return f"Evaluator schema hint unavailable: {type(e).__name__}: {e}"
        return _extract_eval_contract(src, self.expected_output_basename)


def _format_exec_result(result: dict) -> str:
    """Render sandbox dict into a user-facing observation string."""
    out = result.get("stdout", "")
    err = result.get("stderr", "")
    elapsed = result.get("elapsed", 0.0)
    parts = [f"[python_exec] success={result.get('success', False)} elapsed={elapsed:.2f}s"]
    if out:
        parts.append(f"--- stdout ---\n{out.rstrip()}")
    if err:
        parts.append(f"--- stderr ---\n{err.rstrip()}")
    if not out and not err:
        parts.append("(no output)")
    return "\n".join(parts) + "\n"


def _extract_eval_contract(src: str, expected_basename: str | None) -> str | None:
    """Extract non-gold output-format hints from a SAB eval script.

    We intentionally avoid dumping the full evaluator or gold-result paths.
    The goal is to surface schema-level details such as "pd.read_csv" and
    `pred["column"]` references, which explain why many valid-exec files fail
    with pandas/key/length errors.
    """
    if not src:
        return None

    hints: list[str] = []
    read_lines: list[str] = []
    col_names: set[str] = set()

    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line_no_space = line.replace(" ", "")
        mentions_pred_file = "pred_results/" in line
        mentions_expected = bool(expected_basename and expected_basename in line)
        is_output_read = mentions_pred_file or mentions_expected
        if is_output_read and any(fn in line for fn in (
            "read_csv", "read_excel", "read_table", "read_json", "np.load",
            "numpy.load", "Image.open", "open(", "loadmat", "read_h5ad",
        )):
            cleaned = re.sub(r"benchmark/eval_programs/gold_results/[^'\" )]+",
                             "benchmark/.../gold_results/<hidden>", line)
            read_lines.append(cleaned)
        if re.search(r"\b(pred|pred_df|df_pred|y_pred|result)\b", line):
            for m in re.finditer(
                r"(?:pred|pred_df|df_pred|result)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]",
                line,
            ):
                col_names.add(m.group(1))
            for m in re.finditer(
                r"(?:pred|pred_df|df_pred|result)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)",
                line,
            ):
                attr = m.group(1)
                if attr not in {"shape", "values", "columns", "iloc", "loc",
                                "to_numpy", "astype", "squeeze", "head"}:
                    col_names.add(attr)
        # Also catch explicit expected-column lists used in assertions.
        if "columns" in line_no_space and ("[" in line and "]" in line):
            for m in re.finditer(r"['\"]([^'\"]+)['\"]", line):
                val = m.group(1)
                if "/" not in val and "." not in val and len(val) <= 80:
                    col_names.add(val)

    if read_lines:
        hints.append("Evaluator reads the submitted output with:")
        hints.extend(f"  - {line}" for line in read_lines[:6])
        if len(read_lines) > 6:
            hints.append(f"  - ... {len(read_lines) - 6} more read lines omitted")
    if col_names:
        hints.append("Column/name hints referenced by evaluator:")
        hints.append("  - " + ", ".join(sorted(col_names)[:30]))

    if not hints:
        return None
    hints.append(
        "Use these as output-format hints only; still solve the scientific task "
        "from the input data."
    )
    return "\n".join(hints)
