"""Matched SWE repository/tool contract for the three existing code agent loops."""
from .tool_spec import branch_tool, graph_tool, convert_tools_to_description


def create_chat_swe(problem_statement, workflow, *, expose_graph_tools=True):
    tools = [
        {"type": "function", "function": {"name": "python_exec",
         "description": "Execute a fresh Python process in /testbed. Files persist; Python variables do not. "
            "Use pathlib to inspect/edit files and subprocess.run to run shell commands or tests. "
            "The testbed conda environment is active. Network is disabled. Commands time out after 90 seconds; "
            "output retains the beginning and end within 24000 captured bytes. Print results explicitly. "
            "For subprocess.run, print the returncode, stdout and stderr so failures are visible.",
         "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
        {"type": "function", "function": {"name": "finish",
         "description": "Submit repository changes. The evaluator extracts a Git patch; this message is not scored.",
         "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
    ]
    if workflow in ("code_branch", "code_graph"):
        tools += branch_tool()
    elif workflow != "code":
        raise ValueError(f"Unsupported SWE workflow: {workflow}")
    if workflow == "code_graph" and expose_graph_tools:
        tools += graph_tool()
    system = (
        "You are a software engineer fixing the issue below in the Git repository /testbed. "
        "Inspect the relevant code, reproduce the defect, make a minimal correct fix, and run focused tests. "
        "All existing files and dependencies are in the container. Do not fetch external solutions. "
        "Do not alter Git metadata or remove tests to hide a failure. Finish after reviewing your diff. "
        "Only your patch will be evaluated by separate tests.\n\n"
        + convert_tools_to_description(tools)
        + "\nExample inspection:\n<function=python_exec><parameter=code>"
          "import subprocess\nr = subprocess.run(['ls'], capture_output=True, text=True)\n"
          "print('exit_code:', r.returncode)\nprint(r.stdout)\nprint(r.stderr)"
          "</parameter></function>"
    )
    if workflow in ("code_branch", "code_graph"):
        system += (
            "\nYou are MAIN. Delegate focused investigation or implementation using branch when helpful; "
            "then integrate the returned report. Branches have isolated conversation contexts but share "
            "the same repository and filesystem. Delegate sequential, non-conflicting work. "
            "A branch must return its findings; MAIN submits the final patch with finish."
        )
    if workflow == "code_graph":
        system += (
            "\nUse the context graph to retain evidence, file locations, hypotheses, test outcomes, and "
            "branch summaries. Graph edits change memory, not repository files."
        )
        if not expose_graph_tools:
            system += (
                " The controller requests graph edits in [GRAPH ACTION MODE]. Only then emit the "
                "requested JSON. Otherwise use ordinary XML tools."
            )
    return [{"role": "system", "content": system},
            {"role": "user", "content": "# Repository issue\n\n" + problem_statement}]
