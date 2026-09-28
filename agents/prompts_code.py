"""System prompts for code-execution agents (ScienceAgentBench).

Mirrors agents/prompts.py:create_chat() but for `code` / `code_branch` /
`code_graph` workflows. Reuses convert_tools_to_description from
tool_spec.py.
"""

from .tool_spec import convert_tools_to_description
from .tool_spec_code import get_tools_for_workflow


# ── System prompts ──

_CODE_SYSTEM_PROMPT = """You are a data-science agent solving a scientific data-analysis task by executing Python code in a persistent sandbox.

# Workflow
1. **Inspect** the input data files first (shape, columns, dtypes).
2. **Plan** the analytical pipeline mentally before writing code.
3. **Execute** code in small, verifiable steps via `python_exec`. State persists between calls.
4. **Save** your output to the exact path specified in the task instruction (under `pred_results/`).
5. **Verify** that the required output file exists in `pred_results/`.
6. **Finish** when, and ONLY when, the environment reports the output file is present.

# Available tools
{tool_descriptions}

# Tool-call format
Use the XML-tag format. Exactly one tool call per assistant turn. Example:

<function=python_exec>
<parameter=code>
import pandas as pd
df = pd.read_csv("clintox/clintox_train.csv")
print(df.shape, df.columns.tolist())
</parameter>
</function>

# Important rules
- The sandbox preserves state across calls. You do NOT need to re-import or re-load data.
- Each benchmark item is independent. Do NOT reuse code, filenames, paths, variables, or conclusions from another task unless they appear in the current `# Task` or `# Your working environment` block.
- Every data path in your code must be one of the listed input files, a directory containing listed input files, or a file you created earlier in this sandbox.
- Write output files using normal Python (`df.to_csv(...)`, `plt.savefig(...)`). The harness checks `pred_results/` after you `finish`.
- The filename matters. A scientifically reasonable result saved under the wrong name or outside `pred_results/` receives zero credit.
- Do not spend the whole episode inspecting. Use at most two exploratory `python_exec` calls; by the third `python_exec`, create the required output file even if it is only a best-effort solution.
- If the environment says `[output_status] ... present=no`, do not call `finish`; use `python_exec` to create, copy, or rename the required file.
- After creating the final file, run a short verification step that checks `os.path.exists(...)` and prints the file size or directory listing. Call `finish` only after the output status reports the exact required file as present.
- If you cannot fully solve the science task, still write a minimal valid artifact at the exact required path: CSV for `.csv`, JSON for `.json`, PNG via matplotlib for `.png`, text for `.txt`, or the closest requested format. A weak file is better than no file.
- If code raises an exception, the traceback comes back in stderr — read it and fix.
- Do NOT use `os.system`, `subprocess`, shell escapes, or `pip`/`conda` install; they are HARD-BLOCKED by the sandbox and waste a turn. Every scientific package you need (numpy, pandas, scikit-learn, scipy, torch, scanpy, anndata, rdkit, deepchem, DeepPurpose, matplotlib, seaborn, xgboost, statsmodels, geopandas, rasterio, ...) is ALREADY installed — just `import` it.
- Do NOT print the entire dataset — print head/shape/dtypes only.
- Long-running training is OK but each `python_exec` call has a 60-second timeout. Break long training into smaller calls (epochs, etc.) if needed."""


_CODE_BRANCH_ADDENDUM = """

# Branching (sub-task delegation)
You operate as MAIN: plan, delegate, synthesize, and verify. Child agents execute
focused sub-tasks in the same persistent sandbox and return compact reports; their
intermediate stdout is collapsed out of MAIN's context.

- Before MAIN calls `python_exec`, delegate one focused initial branch. Normally
  ask it to inspect the listed inputs, identify schemas/constraints, and propose
  a concrete analysis pipeline. Include the exact output target in its prompt.
- After the branch returns, synthesize its report and continue from MAIN. State
  created by the branch persists, so reuse useful variables and files when safe.
- Use another branch for an independent hypothesis or final verification when it
  materially reduces uncertainty. Branch one task at a time.
- Keep every branch single-purpose and require it to report findings, state
  changes, output paths, failures, and recommended next steps via `return`.
- Do not delegate tiny mechanical steps after the required initial branch; MAIN
  should perform final integration, artifact verification, and `finish`.
"""


_CODE_GRAPH_ADDENDUM = """

# Branching + graph state
On top of branching, you have explicit graph operations to manage your working context:
- `merge` — combine 2+ existing nodes into a single summary node (compress).
- `add_edge` — record a semantic/causal/derivation link between two nodes.
- `select` — set the focus node (next-turn context will be reconstructed around it).
- `prune` — mark a node stale (it stops counting toward your active set).
- `pass` — at a forced consolidation checkpoint, declare you have nothing to consolidate. ONLY valid when the graph is saturated; otherwise penalized.

Use these to keep the active working set small and well-connected. Variables (dataframes, models) you create show up as nodes automatically; you can name and edge them explicitly to track lineage (e.g., `add_edge(adata, X_pca, relation='derived_from')`).
"""


_CODE_GRAPH_CONTROLLER_ADDENDUM = """

# Branching + controller-owned graph state
The harness maintains a ContextGraph from your branch reports and tool observations.
Do not emit graph-management XML actions during normal task execution. A controller
may temporarily enter `[GRAPH ACTION MODE]`; only in that marked mode, return the JSON
object required by the supplied response schema. After `[ENVIRONMENT MODE RESTORED]`,
resume the normal XML protocol using only `python_exec`, `branch`, `return`, or `finish`.

Treat controller checkpoints as part of the scientific workflow, not as a formatting
exercise. Consolidate complementary analyses, connect real data or derivation lineage,
and prune demonstrated dead ends. Use `select` only for a genuine change of analytical
focus, never merely because it is the least destructive choice. Continue the deep
inspect-plan-execute-verify workflow after every checkpoint.
"""


_DISCOVERYBENCH_ADDENDUM = """

# DiscoveryBench result requirements
- Your goal is a defensible natural-language discovery, not a model file or plot.
- Preserve numerical values, comparison groups, direction, functional form,
  thresholds, and boundary conditions supported by the data.
- Save exactly one JSON object to `pred_results/discovery_result.json` with
  non-empty string fields `hypothesis` and `workflow`.
- `hypothesis` must directly answer the discovery question. `workflow` must
  concisely identify the analysis and evidence actually used.
- Do not put Markdown fences around the JSON and do not invent unsupported
  quantitative claims.
"""


_MANIFEST_CAP = 40


def _build_env_block(env) -> str:
    """Render the agent's actual working environment (cwd, real input file
    paths, exact output target) so it does NOT have to guess paths.

    The 8B smoke wasted dozens of turns probing wrong relative paths
    (`atlantic_profiles.nc` -> `ocean_profiles/atlantic_profiles.nc` -> ...)
    because the prompt only carried the CSV folder-tree, not the literal
    layout of files placed in the workdir. This block states the truth.
    """
    if env is None:
        return ""

    workdir = getattr(env, "workdir", None)
    manifest = list(getattr(env, "input_manifest", []) or [])
    expected = getattr(env, "expected_output_basename", None)
    eval_contract = getattr(env, "eval_contract", None)

    lines = ["# Your working environment"]
    if workdir:
        lines.append(
            f"Current working directory: `{workdir}`\n"
            "All relative paths in your code resolve from here."
        )
    if manifest:
        shown = manifest[:_MANIFEST_CAP]
        lines.append(
            "Input files available — open these EXACTLY as written "
            "(paths are relative to the cwd):"
        )
        lines.extend(f"  - {p}" for p in shown)
        if len(manifest) > _MANIFEST_CAP:
            lines.append(f"  - ... and {len(manifest) - _MANIFEST_CAP} more files")
        lines.append(
            "Do NOT guess other paths or probe for files — this list is the "
            "complete, verified set of inputs on disk."
        )
    else:
        lines.append(
            "No input data files were placed in the workdir for this task; "
            "produce the requested output from the instruction alone."
        )
    if expected:
        lines.append(
            "# Output contract\n"
            f"Required final output: `pred_results/{expected}`\n"
            "The pred_results/ directory already exists. The harness scores by "
            "checking that this exact file is present and then running the task "
            "evaluator on it. Before `finish`, the environment must report "
            "`present=yes` for this file.\n"
            "Before `finish`, verify with Python:\n"
            "  import os\n"
            f"  path = 'pred_results/{expected}'\n"
            "  print(path, os.path.exists(path), os.path.getsize(path) if os.path.exists(path) else 'missing')"
        )
    if eval_contract:
        lines.append("# Evaluator output-format hints\n" + eval_contract)
    return "\n".join(lines)


def _build_user_prompt_code(
    instruction: str,
    workflow: str,
    env=None,
    *,
    expose_graph_tools: bool = True,
) -> str:
    tools = get_tools_for_workflow(
        workflow, expose_graph_tools=expose_graph_tools
    )
    tool_desc = convert_tools_to_description(tools)

    sys_prompt = _CODE_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
    if workflow in ('code_branch', 'code_graph'):
        sys_prompt += _CODE_BRANCH_ADDENDUM
    if workflow == 'code_graph':
        sys_prompt += (
            _CODE_GRAPH_ADDENDUM
            if expose_graph_tools
            else _CODE_GRAPH_CONTROLLER_ADDENDUM
        )
    if env is not None and "DiscoveryBench" in str(getattr(env, "ability", "")):
        sys_prompt += _DISCOVERYBENCH_ADDENDUM

    env_block = _build_env_block(env)
    user_msg = f"# Task\n\n{instruction.strip()}\n\n"
    if env_block:
        user_msg += env_block + "\n\n"
    user_msg += (
        "First read this task and the working-environment block carefully, then "
        "inspect only the input files listed above. Do not carry over paths or "
        "code from any previous task. Implement the analysis, save the output "
        "to the exact required path, and verify it exists. If an "
        "`[output_status]` message says the required file is absent, fix that "
        "with `python_exec`. Call `finish` only after the required output is "
        "present."
    )
    if workflow in ("code_branch", "code_graph"):
        user_msg += (
            "\n\nYou are MAIN. Your first tool call must be `branch`, not "
            "`python_exec`. Delegate a focused input-inspection and analysis-plan "
            "task, then synthesize the returned report before continuing."
        )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]


def create_chat_code(
    problem_statement: str,
    workflow: str,
    item=None,
    env=None,
    *,
    expose_graph_tools: bool = True,
):
    """Entry point matching prompts.create_chat signature.

    `env` (optional) is the initialized ScienceAgentEnv; when supplied, the
    user prompt is augmented with the real workdir layout (cwd, input file
    paths, output target) so the agent does not guess paths.
    """
    if getattr(env, "prompt_domain", None) == "swebench":
        from .prompts_swe import create_chat_swe
        return create_chat_swe(
            problem_statement, workflow, expose_graph_tools=expose_graph_tools
        )
    return _build_user_prompt_code(
        problem_statement,
        workflow,
        env=env,
        expose_graph_tools=expose_graph_tools,
    )
