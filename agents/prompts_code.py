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
- Write output files using normal Python (`df.to_csv(...)`, `plt.savefig(...)`). The harness checks `pred_results/` after you `finish`.
- The filename matters. A scientifically reasonable result saved under the wrong name or outside `pred_results/` receives zero credit.
- If the environment says `[output_status] ... present=no`, do not call `finish`; use `python_exec` to create, copy, or rename the required file.
- If code raises an exception, the traceback comes back in stderr — read it and fix.
- Do NOT use `os.system`, `subprocess`, shell escapes, or `pip`/`conda` install; they are HARD-BLOCKED by the sandbox and waste a turn. Every scientific package you need (numpy, pandas, scikit-learn, scipy, torch, scanpy, anndata, rdkit, deepchem, DeepPurpose, matplotlib, seaborn, xgboost, statsmodels, geopandas, rasterio, ...) is ALREADY installed — just `import` it.
- Do NOT print the entire dataset — print head/shape/dtypes only.
- Long-running training is OK but each `python_exec` call has a 60-second timeout. Break long training into smaller calls (epochs, etc.) if needed."""


_CODE_BRANCH_ADDENDUM = """

# Branching (sub-task delegation)
You may delegate a focused sub-task to a child agent via `branch`. Useful when:
- You want to explore data without polluting main context.
- You want to test a hypothesis (e.g., "does featurizer X work?") in isolation.
The child returns a short summary message; its own intermediate stdout is collapsed.

DO NOT branch for trivial sub-steps; the overhead outweighs the benefit. Branch only when the sub-task has clear scope and would generate >5 turns of execution.
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
            "`present=yes` for this file."
        )
    return "\n".join(lines)


def _build_user_prompt_code(instruction: str, workflow: str, env=None) -> str:
    tools = get_tools_for_workflow(workflow)
    tool_desc = convert_tools_to_description(tools)

    sys_prompt = _CODE_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
    if workflow in ('code_branch', 'code_graph'):
        sys_prompt += _CODE_BRANCH_ADDENDUM
    if workflow == 'code_graph':
        sys_prompt += _CODE_GRAPH_ADDENDUM

    env_block = _build_env_block(env)
    user_msg = f"# Task\n\n{instruction.strip()}\n\n"
    if env_block:
        user_msg += env_block + "\n\n"
    user_msg += (
        "Begin by inspecting the input files listed above, then implement the "
        "analysis. Save the output to the exact required path. If an "
        "`[output_status]` message says the required file is absent, fix that "
        "with `python_exec`. Call `finish` only after the required output is "
        "present."
    )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]


def create_chat_code(problem_statement: str, workflow: str, item=None, env=None):
    """Entry point matching prompts.create_chat signature.

    `env` (optional) is the initialized ScienceAgentEnv; when supplied, the
    user prompt is augmented with the real workdir layout (cwd, input file
    paths, output target) so the agent does not guess paths.
    """
    return _build_user_prompt_code(problem_statement, workflow, env=env)
