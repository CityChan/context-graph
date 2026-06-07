"""Tool spec for code-execution agents (ScienceAgentBench).

Mirrors agents/tool_spec.py but the ONLY external tool is python_exec.
Branch / merge / add_edge / select / prune / return / finish are reused
unchanged from tool_spec.py (re-exported for convenience).

Workflow names follow the search_* convention:
  - code            : single-thread ReAct (python_exec + finish only)
  - code_branch     : fold-agent style (adds branch + return)
  - code_graph      : ContextGraph (adds merge/add_edge/select/prune)
"""

from .tool_spec import branch_tool, graph_tool, convert_tools_to_description


def python_exec_tool() -> dict:
    return {
        'type': 'function',
        'function': {
            'name': 'python_exec',
            'description': (
                "Execute Python code in a persistent sandbox. State (imported "
                "modules, defined variables, loaded data) PERSISTS across calls "
                "within the same trajectory. Returns stdout + stderr (each "
                "truncated to 2KB). On exception, the traceback is returned in "
                "stderr; the trajectory continues so you can debug.\n"
                "Working directory contains the task's input data files; you "
                "MUST write your output files to the `pred_results/` subdirectory "
                "(the directory is created for you).\n"
                "Long-running cells (>60s) will time out and return a partial "
                "stdout."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': (
                            "Python code to execute. Multi-line is fine. "
                            "Imports persist. Variables persist. Use `print(...)` "
                            "to surface values; the return value of the last "
                            "expression is NOT auto-printed."
                        ),
                    },
                },
                'required': ['code'],
            },
        },
    }


def finish_tool() -> dict:
    """Signal completion. The 'message' arg is logged; reward is computed by
    inspecting files written to pred_results/, not by parsing this message."""
    return {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': (
                "Signal that you have completed the task. The scoring is based "
                "on the file(s) you wrote to pred_results/, NOT on this message. "
                "Only call this AFTER your output file is on disk."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'description': "Brief summary of what you produced.",
                    },
                },
                'required': ['message'],
            },
        },
    }


# ── Workflow assembly ──

def get_tools_for_workflow(workflow: str) -> list[dict]:
    """Return the tool list a given workflow exposes to the LLM."""
    base = [python_exec_tool(), finish_tool()]
    if workflow == 'code':
        return base
    if workflow == 'code_branch':
        return base + [branch_tool()]
    if workflow == 'code_graph':
        return base + [branch_tool(), graph_tool()]
    raise ValueError(f"Unknown code workflow: {workflow}")
