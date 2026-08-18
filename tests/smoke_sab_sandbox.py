"""Standalone smoke test for ScienceAgentBench sandbox + env.

Runs LOCALLY without verl, vLLM, Ray, or Vista. Validates:

  1. CodeSandbox: stateful exec, stdout capture, timeout, error reporting.
  2. ScienceAgentEnv: init_env sets up workdir, run_action dispatches
     python_exec, finish wraps up, get_reward grades file presence.
  3. The XML tool-call format we expect from the LLM (<function=...>) is
     parsed correctly by envs.local_search.extract_fn_call.

Run:
    python -m tests.smoke_sab_sandbox

Pass criteria printed at the end. Exit code 0 = all pass, 1 = any fail.
"""

from __future__ import annotations

import asyncio
import os
import sys
import shutil
import tempfile
import textwrap
from types import SimpleNamespace


# ── Test 1: CodeSandbox — stateful exec ──

def test_sandbox_basic():
    from envs.scienceagent_sandbox import CodeSandbox

    with tempfile.TemporaryDirectory() as tmp:
        sb = CodeSandbox(tmp, per_call_timeout=5.0)

        # Call 1: define a variable
        r1 = sb.execute("x = 41\nprint('hello')")
        assert r1['success'] is True, f"call 1 failed: {r1!r}"
        assert 'hello' in r1['stdout'], f"call 1 stdout: {r1['stdout']!r}"

        # Call 2: variable from call 1 must persist
        r2 = sb.execute("print(x + 1)")
        assert r2['success'] is True, f"call 2 failed: {r2!r}"
        assert '42' in r2['stdout'], f"call 2 stdout: {r2['stdout']!r}"

        # Call 3: error should be returned in stderr, not propagated
        r3 = sb.execute("raise ValueError('boom')")
        assert r3['success'] is False, f"call 3 should report failure"
        assert 'ValueError' in r3['stderr'], f"call 3 stderr: {r3['stderr']!r}"
        assert 'boom' in r3['stderr'], f"call 3 stderr: {r3['stderr']!r}"

        # Call 4: state still intact after error
        r4 = sb.execute("print(x * 2)")
        assert r4['success'] is True
        assert '82' in r4['stdout']

        sb.close()
    return "test_sandbox_basic"


def test_sandbox_timeout():
    """Timeout via signal.SIGALRM is POSIX-only and main-thread-only.
    On Windows or from a background thread, the sandbox is best-effort no-timeout
    so the call sleeps the full 5s. We skip the strict assertion on Windows.
    """
    import signal
    from envs.scienceagent_sandbox import CodeSandbox

    if not hasattr(signal, "SIGALRM"):
        return "test_sandbox_timeout (SKIPPED — no SIGALRM on this platform)"

    with tempfile.TemporaryDirectory() as tmp:
        sb = CodeSandbox(tmp, per_call_timeout=1.0)
        r = sb.execute("import time\ntime.sleep(5)")
        assert r['success'] is False, f"should have timed out, got {r!r}"
        assert 'timed out' in r['stderr'].lower(), f"stderr: {r['stderr']!r}"
        assert r['elapsed'] < 2.0, f"should not have slept 5s, elapsed={r['elapsed']}"
        sb.close()
    return "test_sandbox_timeout"


def test_sandbox_pred_results_dir():
    from envs.scienceagent_sandbox import CodeSandbox

    with tempfile.TemporaryDirectory() as tmp:
        sb = CodeSandbox(tmp, per_call_timeout=5.0)

        # pred_results/ must be auto-created
        assert os.path.isdir(os.path.join(tmp, "pred_results"))

        # Agent writes a file
        code = textwrap.dedent("""
            import os, json
            os.makedirs('pred_results', exist_ok=True)
            with open('pred_results/result.csv', 'w') as f:
                f.write('col1,col2\\n1,2\\n3,4\\n')
        """).strip()
        r = sb.execute(code)
        assert r['success'] is True, f"write failed: {r!r}"

        files = sb.list_output_files()
        assert files == ['result.csv'], f"got files: {files}"
        sb.close()
    return "test_sandbox_pred_results_dir"


def test_sandbox_blocks_shell_and_pip():
    """Shell escapes / pip-install / subprocess must be rejected before exec
    with a corrective message (the first smoke saw an agent run
    `pip install DeepPurpose` in-sandbox). Legit code must still run."""
    from envs.scienceagent_sandbox import CodeSandbox

    with tempfile.TemporaryDirectory() as tmp:
        sb = CodeSandbox(tmp, per_call_timeout=5.0)

        blocked_snippets = [
            "import subprocess; subprocess.run(['pip', 'install', 'DeepPurpose'])",
            "import os; os.system('pip install scanpy')",
            "import sys, subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rdkit'])",
            "!pip install deepchem",
            "%pip install torch",
            "import os; os.popen('ls')",
        ]
        for snip in blocked_snippets:
            r = sb.execute(snip)
            assert r['success'] is False, f"should be blocked: {snip!r} -> {r!r}"
            assert 'Blocked' in r['stderr'], f"no block msg for {snip!r}: {r['stderr']!r}"
            assert r['elapsed'] == 0.0, f"blocked call should not execute: {r!r}"

        # Legit scientific code still runs, and state is untouched by the blocks.
        r_ok = sb.execute("import math; print(math.sqrt(16))")
        assert r_ok['success'] is True, f"legit code blocked: {r_ok!r}"
        assert '4.0' in r_ok['stdout'], f"stdout: {r_ok['stdout']!r}"

        sb.close()
    return "test_sandbox_blocks_shell_and_pip"


def test_prewarm_idempotent():
    """prewarm_heavy_imports must run without raising even when packages are
    absent (they're skipped), and be a no-op on the second call. On this dev
    box most heavy packages are missing — that's fine, we only assert it does
    not throw and flips the done-flag."""
    import envs.scienceagent_sandbox as sb_mod

    # Reset the guard so the test exercises the real import loop once.
    sb_mod._PREWARM_DONE = False
    sb_mod.prewarm_heavy_imports(verbose=False)
    assert sb_mod._PREWARM_DONE is True, "prewarm should set the done flag"

    # Second call is a no-op (must not raise, must not redo work).
    sb_mod.prewarm_heavy_imports(verbose=False)
    assert sb_mod._PREWARM_DONE is True
    return "test_prewarm_idempotent"


def test_prewarm_package_override():
    """Benchmark wrappers can exclude native imports that are unsafe at startup."""
    import envs.scienceagent_sandbox as sb_mod

    previous = os.environ.get("SAB_PREWARM_PACKAGES")
    try:
        os.environ["SAB_PREWARM_PACKAGES"] = "numpy, pandas,,scipy "
        assert sb_mod._configured_prewarm_packages() == ["numpy", "pandas", "scipy"]
        os.environ["SAB_PREWARM_PACKAGES"] = ""
        assert sb_mod._configured_prewarm_packages() == []
    finally:
        if previous is None:
            os.environ.pop("SAB_PREWARM_PACKAGES", None)
        else:
            os.environ["SAB_PREWARM_PACKAGES"] = previous
    return "test_prewarm_package_override"


# ── Test 1b: prompt builders + workflow tool assembly ──

def test_prompt_code_workflow_assembly():
    from agents.tool_spec_code import get_tools_for_workflow
    from agents.prompts_code import create_chat_code

    # code (ReAct): python_exec + finish only
    tools = get_tools_for_workflow('code')
    names = [t['function']['name'] for t in tools]
    assert names == ['python_exec', 'finish'], f"code tools: {names}"

    # code_branch: add branch
    tools_b = get_tools_for_workflow('code_branch')
    names_b = [t['function']['name'] for t in tools_b]
    assert 'branch' in names_b, f"code_branch tools: {names_b}"
    assert 'python_exec' in names_b
    assert 'merge' not in names_b, "code_branch should NOT expose merge"

    # code_graph: add branch + merge + add_edge + select + prune
    tools_g = get_tools_for_workflow('code_graph')
    names_g = [t['function']['name'] for t in tools_g]
    assert all(n in names_g for n in ['python_exec', 'branch', 'merge', 'add_edge', 'select', 'prune']), \
        f"code_graph tools: {names_g}"

    # Prompt builder includes the python_exec tool description
    chat = create_chat_code("Compute the mean of column 'value'.", 'code')
    assert isinstance(chat, list) and len(chat) == 2
    assert chat[0]['role'] == 'system' and chat[1]['role'] == 'user'
    assert 'python_exec' in chat[0]['content']
    assert 'persistent sandbox' in chat[0]['content']
    assert 'Each benchmark item is independent' in chat[0]['content']
    assert 'Compute the mean' in chat[1]['content']

    env = SimpleNamespace(
        workdir="/tmp/sab-task",
        input_manifest=["input.csv"],
        expected_output_basename="result.csv",
        eval_contract=None,
    )
    chat_env = create_chat_code("Compute the mean of column 'value'.", 'code', env=env)
    prompt = chat_env[1]['content']
    assert 'Do not carry over paths or code from any previous task' in prompt
    assert "path = 'pred_results/result.csv'" in prompt
    assert 'input.csv' in prompt

    # code_graph prompt includes the graph addendum
    chat_g = create_chat_code("foo", 'code_graph')
    assert 'merge' in chat_g[0]['content'].lower()
    assert 'pass' in chat_g[0]['content'].lower()  # saturation pass hint
    return "test_prompt_code_workflow_assembly"


# ── Test 2: extract_fn_call (existing utility) parses XML format we emit ──

def test_xml_tool_call_parsing():
    from envs.local_search import extract_fn_call

    response = textwrap.dedent("""
        Looking at the task, I'll inspect the data first.

        <function=python_exec>
        <parameter=code>
        import pandas as pd
        df = pd.read_csv('data.csv')
        print(df.shape)
        </parameter>
        </function>
    """).strip()
    calls = extract_fn_call(response)
    assert calls is not None and len(calls) == 1, f"calls: {calls}"
    assert calls[0]['function'] == 'python_exec'
    assert 'pd.read_csv' in calls[0]['arguments']['code']

    finish_resp = textwrap.dedent("""
        <function=finish>
        <parameter=message>saved result.csv</parameter>
        </function>
    """).strip()
    calls = extract_fn_call(finish_resp)
    assert calls[0]['function'] == 'finish'
    assert calls[0]['arguments']['message'] == 'saved result.csv'
    return "test_xml_tool_call_parsing"


# ── Test 2b: CSV loader produces task dicts the env can consume ──

_FAKE_CSV_HEADER = (
    "instance_id,domain,subtask_categories,github_name,task_inst,"
    "domain_knowledge,dataset_folder_tree,dataset_preview,"
    "src_file_or_path,gold_program_name,output_fname,eval_script_name\n"
)


def _write_fake_sab_csv(path: str) -> None:
    rows = [
        # Minimal first task — uses synthetic input.csv we will create in workdir
        (
            "1,Computational Chemistry,\"Feature Engineering\","
            "deepchem/deepchem,"
            "\"Compute the mean of the 'value' column of input.csv and save "
            "as pred_results/result.csv with column 'mean_value'.\","
            "\"This is a smoke task. Use pandas.\","
            "\"|-- input.csv\","
            "\"name,value\\nalice,10\\nbob,20\\ncarol,30\","
            "examples/smoke,smoke_program.py,"
            "pred_results/result.csv,eval_smoke.py\n"
        ),
        # Second task with different output path
        (
            "2,Bioinformatics,\"Visualization\","
            "scverse/scvi-tutorials,"
            "\"Save a CSV with single row count=99 to pred_results/count.csv\","
            "\"\","
            "\"\","
            "\"\","
            "examples/count,count.py,"
            "pred_results/count.csv,eval_count.py\n"
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(_FAKE_CSV_HEADER)
        for row in rows:
            f.write(row)


def test_loader_basic():
    from envs.scienceagent_loader import load_sab_tasks

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sab.csv")
        _write_fake_sab_csv(csv_path)

        tasks = load_sab_tasks(csv_path, benchmark_dir=None, workflow='code_branch')
        assert len(tasks) == 2, f"got {len(tasks)} tasks"

        t1 = tasks[0]
        assert t1['task_id'] == '1'
        assert t1['expected_output'] == 'pred_results/result.csv'
        assert t1['workflow'] == 'code_branch'
        assert t1['input_files'] == [], "no benchmark_dir -> empty input_files"
        # Composite instruction should include domain knowledge + preview
        assert 'Compute the mean' in t1['instruction']
        assert 'Domain Knowledge' in t1['instruction']
        assert 'Dataset Folder Tree' in t1['instruction']
        assert 'Dataset Preview' in t1['instruction']
        assert 'alice' in t1['instruction']  # preview content

        # Subset filter
        only_2 = load_sab_tasks(csv_path, instance_ids=[2])
        assert len(only_2) == 1 and only_2[0]['task_id'] == '2'

    return "test_loader_basic"


def test_eval_imports_project_support_module():
    """Eval subprocesses can import helpers shipped at the project root."""
    from envs.scienceagent_eval import score_task

    with tempfile.TemporaryDirectory() as tmp:
        benchmark_dir = os.path.join(tmp, "benchmark")
        eval_dir = os.path.join(benchmark_dir, "eval_programs")
        workdir = os.path.join(tmp, "workdir")
        os.makedirs(eval_dir)
        os.makedirs(os.path.join(workdir, "pred_results"))
        with open(os.path.join(workdir, "pred_results", "plot.png"), "wb") as f:
            f.write(b"not-a-real-png")
        with open(os.path.join(eval_dir, "eval_visual.py"), "w") as f:
            f.write(
                "from gpt4_visual_judge import encode_image\n"
                "print((int(callable(encode_image)), 'helper imported'))\n"
            )

        result = score_task(
            workdir,
            benchmark_dir,
            "eval_visual.py",
            "pred_results/plot.png",
        )
        assert result["score"] == 1.0, result
        assert result["rule"] == "tuple", result

    return "test_eval_imports_project_support_module"


def test_eval_resolves_relative_benchmark_path():
    """Changing evaluator cwd must not reinterpret a relative benchmark path."""
    from envs.scienceagent_eval import score_task

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        benchmark_dir = os.path.join(tmp, "benchmark")
        eval_dir = os.path.join(benchmark_dir, "eval_programs")
        workdir = os.path.join(tmp, "workdir")
        os.makedirs(eval_dir)
        os.makedirs(os.path.join(workdir, "pred_results"))
        with open(os.path.join(workdir, "pred_results", "result.txt"), "w") as f:
            f.write("ok")
        with open(os.path.join(eval_dir, "eval_relative.py"), "w") as f:
            f.write("print((1, 'relative path resolved'))\n")

        result = score_task(
            os.path.relpath(workdir),
            os.path.relpath(benchmark_dir),
            "eval_relative.py",
            "pred_results/result.txt",
        )
        assert result["score"] == 1.0, result
        assert result["rule"] == "tuple", result

    return "test_eval_resolves_relative_benchmark_path"


async def test_prompt_eval_contract_is_gated():
    """Eval-script schema hints are useful diagnostics but must be opt-in."""
    from agents.prompts_code import create_chat_code
    from envs.scienceagent_loader import load_sab_tasks
    from envs.scienceagent_env import ScienceAgentEnv

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sab.csv")
        _write_fake_sab_csv(csv_path)

        bench_dir = os.path.join(tmp, "benchmark")
        os.makedirs(os.path.join(bench_dir, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(bench_dir, "eval_programs"), exist_ok=True)
        with open(os.path.join(bench_dir, "datasets", "input.csv"), "w") as f:
            f.write("name,value\nalice,10\nbob,20\ncarol,30\n")
        with open(os.path.join(bench_dir, "eval_programs", "eval_smoke.py"), "w") as f:
            f.write(
                "import pandas as pd\n"
                "def eval():\n"
                "    pred = pd.read_csv('pred_results/result.csv')\n"
                "    return int(pred['mean_value'].iloc[0] == 20), 'ok'\n"
            )

        tasks = load_sab_tasks(csv_path, benchmark_dir=bench_dir, workflow='code')
        task_dict = dict(tasks[0])
        task_dict['workdir'] = os.path.join(tmp, "run")
        item = _MockDataProto(task_dict, task_dict['instruction'])
        config = SimpleNamespace(plugin=SimpleNamespace(sandbox_timeout=10.0))

        env = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
        old = os.environ.pop("SAB_EXPOSE_EVAL_CONTRACT", None)
        try:
            await env.init_env(item)
            chat = create_chat_code(task_dict['instruction'], 'code', env=env)
            assert 'Evaluator output-format hints' not in chat[1]['content']
            env.close()

            os.environ["SAB_EXPOSE_EVAL_CONTRACT"] = "1"
            env2 = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
            await env2.init_env(item)
            chat2 = create_chat_code(task_dict['instruction'], 'code', env=env2)
            prompt = chat2[1]['content']
            assert 'Evaluator output-format hints' in prompt, prompt
            assert "pd.read_csv('pred_results/result.csv')" in prompt, prompt
            assert "mean_value" in prompt, prompt
            env2.close()
        finally:
            if old is None:
                os.environ.pop("SAB_EXPOSE_EVAL_CONTRACT", None)
            else:
                os.environ["SAB_EXPOSE_EVAL_CONTRACT"] = old

    return "test_prompt_eval_contract_is_gated"


async def test_interactive_eval_feedback_is_gated():
    """Interactive evaluator feedback is opt-in and appears after output exists."""
    from envs.scienceagent_loader import load_sab_tasks
    from envs.scienceagent_env import ScienceAgentEnv

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sab.csv")
        _write_fake_sab_csv(csv_path)

        bench_dir = os.path.join(tmp, "benchmark")
        os.makedirs(os.path.join(bench_dir, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(bench_dir, "eval_programs"), exist_ok=True)
        with open(os.path.join(bench_dir, "datasets", "input.csv"), "w") as f:
            f.write("name,value\nalice,10\nbob,20\ncarol,30\n")
        with open(os.path.join(bench_dir, "eval_programs", "eval_smoke.py"), "w") as f:
            f.write(
                "import pandas as pd\n"
                "def eval():\n"
                "    pred = pd.read_csv('pred_results/result.csv')\n"
                "    return int(pred['mean_value'].iloc[0] == 20), 'ok'\n"
                "if __name__ == '__main__':\n"
                "    print(eval())\n"
            )

        tasks = load_sab_tasks(csv_path, benchmark_dir=bench_dir, workflow='code')
        task_dict = dict(tasks[0])
        task_dict['workdir'] = os.path.join(tmp, "run")
        item = _MockDataProto(task_dict, task_dict['instruction'])
        config = SimpleNamespace(plugin=SimpleNamespace(sandbox_timeout=10.0))

        old = os.environ.pop("SAB_INTERACTIVE_EVAL_FEEDBACK", None)
        try:
            env = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
            await env.init_env(item)
            bad_resp = textwrap.dedent("""
                <function=python_exec>
                <parameter=code>
                import pandas as pd
                pd.DataFrame({'wrong_col': [20]}).to_csv('pred_results/result.csv', index=False)
                print('wrote bad schema')
                </parameter>
                </function>
            """).strip()
            out = await env.run_action(bad_resp)
            assert '[eval_feedback]' not in out['observation'], out['observation']
            env.close()

            task_dict['workdir'] = os.path.join(tmp, "run2")
            item2 = _MockDataProto(task_dict, task_dict['instruction'])
            os.environ["SAB_INTERACTIVE_EVAL_FEEDBACK"] = "1"
            env2 = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
            await env2.init_env(item2)
            out2 = await env2.run_action(bad_resp)
            assert '[eval_feedback]' in out2['observation'], out2['observation']
            assert 'score=0.000' in out2['observation'], out2['observation']
            assert 'Fix the output file' in out2['observation'], out2['observation']

            fix_resp = textwrap.dedent("""
                <function=python_exec>
                <parameter=code>
                import pandas as pd
                pd.DataFrame({'mean_value': [20]}).to_csv('pred_results/result.csv', index=False)
                print('fixed schema')
                </parameter>
                </function>
            """).strip()
            out3 = await env2.run_action(fix_resp)
            assert '[eval_feedback]' in out3['observation'], out3['observation']
            assert 'score=1.000' in out3['observation'], out3['observation']
            assert 'may call finish' in out3['observation'], out3['observation']
            env2.close()
        finally:
            if old is None:
                os.environ.pop("SAB_INTERACTIVE_EVAL_FEEDBACK", None)
            else:
                os.environ["SAB_INTERACTIVE_EVAL_FEEDBACK"] = old

    return "test_interactive_eval_feedback_is_gated"


def test_loader_deep_nested_tree():
    """SAB has tasks with 4-5 level nested trees (e.g., BBBC002 image
    directories, RGI60 glacier archives). Earlier loader only handled
    2 levels, dropping 6/102 tasks. Regression guard."""
    from envs.scienceagent_loader import load_sab_tasks

    with tempfile.TemporaryDirectory() as tmp:
        # Build a real 4-level nested input file
        bench = os.path.join(tmp, "bench")
        deep = os.path.join(bench, "datasets", "BBBC002", "test",
                            "drosophila_kc167_1_images")
        os.makedirs(deep, exist_ok=True)
        leaf_file = os.path.join(deep, "CPvalid1_340_40x_Tiles_p1175DAPI.TIF")
        with open(leaf_file, "w") as f:
            f.write("(fake)")

        # CSV row pointing at this 4-level tree. Real SAB embeds REAL newlines
        # inside the quoted `dataset_folder_tree` field (CSV-legal), so we write
        # actual \n characters here, not literal backslash-n.
        tree = (
            "|-- BBBC002/\n"
            "|---- test/\n"
            "|------ drosophila_kc167_1_images/\n"
            "|-------- CPvalid1_340_40x_Tiles_p1175DAPI.TIF"
        )
        csv_path = os.path.join(tmp, "sab.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(_FAKE_CSV_HEADER)
            f.write(
                f"11,Bioinformatics,\"Image Analysis\",broad-institute/BBBC,"
                f"\"Train an image classifier on BBBC002 cell images.\","
                f"\"BBBC002 is fluorescence microscopy.\","
                f"\"{tree}\","
                f"\"\",examples/bbbc002,bbbc002.py,"
                f"pred_results/bbbc002.csv,eval_bbbc002.py\n"
            )

        tasks = load_sab_tasks(csv_path, benchmark_dir=bench)
        assert len(tasks) == 1, f"got {len(tasks)} tasks"
        ifs = tasks[0]['input_files']
        assert len(ifs) == 1, f"deep tree returned {len(ifs)} files: {ifs}"
        nested = os.path.join("BBBC002", "test", "drosophila_kc167_1_images",
                              "CPvalid1_340_40x_Tiles_p1175DAPI.TIF")
        assert ifs[0].endswith(nested), f"wrong abs path: {ifs[0]}"
        # The rel path (what the env must reproduce under workdir) is carried too.
        rels = tasks[0]['input_rel_paths']
        assert rels == [nested], f"wrong rel paths: {rels}"
    return "test_loader_deep_nested_tree"


async def test_loader_with_env():
    """End-to-end: loader -> env.init_env -> python_exec -> finish."""
    from envs.scienceagent_loader import load_sab_tasks
    from envs.scienceagent_env import ScienceAgentEnv

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sab.csv")
        _write_fake_sab_csv(csv_path)

        # Fake "benchmark_dir" with the input file the first task expects
        bench_dir = os.path.join(tmp, "benchmark")
        os.makedirs(os.path.join(bench_dir, "datasets"), exist_ok=True)
        with open(os.path.join(bench_dir, "datasets", "input.csv"), "w") as f:
            f.write("name,value\nalice,10\nbob,20\ncarol,30\n")

        tasks = load_sab_tasks(csv_path, benchmark_dir=bench_dir, workflow='code')
        assert len(tasks[0]['input_files']) == 1, f"input files: {tasks[0]['input_files']}"

        # Now wire to env
        config = SimpleNamespace(plugin=SimpleNamespace(sandbox_timeout=10.0))
        env = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
        # ScienceAgentEnv.init_env reads .item() then dispatches
        task_dict = dict(tasks[0])
        task_dict['workdir'] = os.path.join(tmp, "run_workdir")
        item = _MockDataProto(task_dict, task_dict['instruction'])
        await env.init_env(item)

        # Verify input file was symlinked/copied into workdir
        assert os.path.exists(os.path.join(env.workdir, "input.csv")), \
            f"workdir contents: {os.listdir(env.workdir)}"

        # Trivial agent: compute mean + save
        resp = textwrap.dedent("""
            <function=python_exec>
            <parameter=code>
            import pandas as pd, os
            df = pd.read_csv('input.csv')
            os.makedirs('pred_results', exist_ok=True)
            pd.DataFrame({'mean_value': [df['value'].mean()]}).to_csv(
                'pred_results/result.csv', index=False)
            print('done')
            </parameter>
            </function>
        """).strip()
        out = await env.run_action(resp)
        assert 'done' in out['observation']

        fin = textwrap.dedent("""
            <function=finish>
            <parameter=message>saved</parameter>
            </function>
        """).strip()
        out2 = await env.run_action(fin)
        assert out2.get('action') == 'finish'

        score_msg, reward, info = await env.get_reward(item, [], None)
        assert reward == 1.0, f"reward: {reward}, msg: {score_msg}"

        env.close()
    return "test_loader_with_env"


# ── Test 3: ScienceAgentEnv end-to-end on a synthetic task ──

class _MockExtraInfo(dict):
    """Mimic verl's non_tensor_batch entry: an object with .ndim/.item() shape."""
    @property
    def ndim(self):
        return 0
    def item(self):
        return self


class _MockDataProto:
    def __init__(self, task_dict: dict, problem_statement: str, ability: str = 'ScienceAgentBench'):
        self.non_tensor_batch = {
            'ability': _MockExtraInfo({'_': ability}),
            'extra_info': _MockExtraInfo(task_dict),
        }
    meta_info: dict = {}


def _make_synthetic_task(workdir: str) -> dict:
    """Generate a minimal CSV input and define the task spec."""
    csv_path = os.path.join(workdir, "input_data.csv")
    with open(csv_path, "w") as f:
        f.write("name,value\nalice,10\nbob,20\ncarol,30\n")
    return {
        'task_id': 'smoke_001',
        'instruction': (
            "Compute the mean of the 'value' column in input_data.csv and "
            "save it as a single-row CSV with column 'mean_value' to "
            "pred_results/result.csv."
        ),
        'input_files': [csv_path],
        'expected_output': 'pred_results/result.csv',
        'workflow': 'code',
        'workdir': workdir + "_sandbox",  # let env create a fresh dir for the run
    }


async def test_env_end_to_end():
    from envs.scienceagent_env import ScienceAgentEnv

    with tempfile.TemporaryDirectory() as base:
        task_dict = _make_synthetic_task(base)

        # Mock config: just needs config.plugin.sandbox_timeout
        config = SimpleNamespace(plugin=SimpleNamespace(sandbox_timeout=10.0))

        env = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
        item = _MockDataProto(task_dict, task_dict['instruction'])
        await env.init_env(item)

        # Verify input was copied/symlinked into workdir
        assert os.path.exists(os.path.join(env.workdir, 'input_data.csv')), \
            f"input file not in workdir {env.workdir}"
        assert env.instance_info['problem_statement'].startswith("Compute the mean")

        # Turn 1: agent emits python_exec to inspect data
        resp1 = textwrap.dedent("""
            <function=python_exec>
            <parameter=code>
            import pandas as pd
            df = pd.read_csv('input_data.csv')
            print(df.shape, df.columns.tolist())
            </parameter>
            </function>
        """).strip()
        out1 = await env.run_action(resp1)
        assert 'observation' in out1, f"out1: {out1!r}"
        assert '(3, 2)' in out1['observation'], f"out1 observation: {out1['observation']!r}"
        assert 'present=no' in out1['observation'], f"out1 observation: {out1['observation']!r}"

        # Turn 2: compute + write
        resp2 = textwrap.dedent("""
            <function=python_exec>
            <parameter=code>
            import os
            mean_val = df['value'].mean()
            print('mean:', mean_val)
            os.makedirs('pred_results', exist_ok=True)
            pd.DataFrame({'mean_value': [mean_val]}).to_csv('pred_results/result.csv', index=False)
            </parameter>
            </function>
        """).strip()
        out2 = await env.run_action(resp2)
        assert 'mean: 20' in out2['observation']
        assert 'present=yes' in out2['observation'], f"out2 observation: {out2['observation']!r}"

        # Turn 3: finish
        resp3 = textwrap.dedent("""
            <function=finish>
            <parameter=message>saved pred_results/result.csv with mean=20</parameter>
            </function>
        """).strip()
        out3 = await env.run_action(resp3)
        assert out3.get('action') == 'finish', f"out3: {out3}"

        # Reward: expected_output is present
        score_msg, reward, reward_dict = await env.get_reward(item, messages=[], context=None)
        assert reward == 1.0, f"reward: {reward}, msg: {score_msg}"

        # Stats
        assert env.stats['python_exec'] == 2
        assert env.stats['finish'] == 1
        assert env.stats['is_finish'] == 1

        env.close()
    return "test_env_end_to_end"


async def test_env_preserves_input_subdir():
    """Regression: the env must place input files at workdir/<rel> preserving
    subdirs (e.g. dkpes/dkpes_train.csv), NOT flatten to the basename — the
    first 4-node smoke showed agents doing open('dkpes/dkpes_train.csv') and
    getting FileNotFound because the env had symlinked it to workdir/
    dkpes_train.csv."""
    from envs.scienceagent_env import ScienceAgentEnv

    with tempfile.TemporaryDirectory() as base:
        # A source data file living under a subdir, like the real benchmark.
        src = os.path.join(base, "datasets", "dkpes", "dkpes_train.csv")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        with open(src, "w") as f:
            f.write("smiles,label\nCCO,1\n")

        task_dict = {
            'task_id': 'subdir_001',
            'instruction': "Read dkpes/dkpes_train.csv and save pred_results/out.csv.",
            'input_files': [src],
            'input_rel_paths': [os.path.join("dkpes", "dkpes_train.csv")],
            'expected_output': 'pred_results/out.csv',
            'workflow': 'code',
            'workdir': os.path.join(base, "run"),
        }
        config = SimpleNamespace(plugin=SimpleNamespace(sandbox_timeout=10.0))
        env = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
        item = _MockDataProto(task_dict, task_dict['instruction'])
        await env.init_env(item)

        # The file is reachable at the EXACT relative path the agent will use.
        assert os.path.exists(os.path.join(env.workdir, "dkpes", "dkpes_train.csv")), \
            f"subdir not preserved; workdir tree: {os.listdir(env.workdir)}"

        # And the agent can actually open it with the relative path.
        resp = textwrap.dedent("""
            <function=python_exec>
            <parameter=code>
            import pandas as pd, os
            df = pd.read_csv('dkpes/dkpes_train.csv')
            os.makedirs('pred_results', exist_ok=True)
            df.to_csv('pred_results/out.csv', index=False)
            print('rows', len(df))
            </parameter>
            </function>
        """).strip()
        out = await env.run_action(resp)
        assert 'rows 1' in out['observation'], f"obs: {out['observation']!r}"

        score_msg, reward, _ = await env.get_reward(item, [], None)
        assert reward == 1.0, f"reward {reward}: {score_msg}"
        env.close()
    return "test_env_preserves_input_subdir"


async def test_env_missing_output():
    """Finish is rejected, and reward remains 0, if expected file is absent."""
    from envs.scienceagent_env import ScienceAgentEnv

    with tempfile.TemporaryDirectory() as base:
        task_dict = _make_synthetic_task(base)
        config = SimpleNamespace(plugin=SimpleNamespace(sandbox_timeout=10.0))
        env = ScienceAgentEnv(config, tokenizer=None, ability='ScienceAgentBench')
        item = _MockDataProto(task_dict, task_dict['instruction'])
        await env.init_env(item)

        # Agent emits python_exec that writes to WRONG path, then finishes.
        resp1 = textwrap.dedent("""
            <function=python_exec>
            <parameter=code>
            with open('wrong_path.csv', 'w') as f:
                f.write('mean_value\\n20\\n')
            print('done')
            </parameter>
            </function>
        """).strip()
        out1 = await env.run_action(resp1)
        assert '[progress_hint]' not in out1['observation'], out1['observation']
        await env.run_action(resp1)
        out_hint = await env.run_action(resp1)
        assert '[progress_hint]' in out_hint['observation'], out_hint['observation']
        assert 'write a best-effort output now' in out_hint['observation'], out_hint['observation']

        resp2 = textwrap.dedent("""
            <function=finish>
            <parameter=message>oops, wrong path</parameter>
            </function>
        """).strip()
        out2 = await env.run_action(resp2)
        assert 'observation' in out2, f"out2: {out2!r}"
        assert '[finish rejected]' in out2['observation'], f"obs: {out2['observation']!r}"
        assert 'present=no' in out2['observation'], f"obs: {out2['observation']!r}"
        assert env.stats['finish'] == 0
        assert env.stats['finish_rejected'] == 1

        score_msg, reward, info = await env.get_reward(item, messages=[], context=None)
        assert reward == 0.0, f"reward should be 0 (file missing), got {reward}"
        # When NO files are produced in pred_results/, env returns "no output files"
        # with produced_files=0. When the wrong file is produced, expected_missing
        # would be set. Here the agent wrote to wrong_path.csv (workdir root, not
        # pred_results/), so produced_files=0.
        assert info.get('produced_files') == 0, f"info: {info}"
        assert 'no output' in score_msg, f"score_msg: {score_msg}"
        env.close()
    return "test_env_missing_output"


# ── Runner ──

async def main():
    results = []
    failures = 0
    sync_tests = [
        test_sandbox_basic,
        test_sandbox_timeout,
        test_sandbox_pred_results_dir,
        test_sandbox_blocks_shell_and_pip,
        test_prewarm_idempotent,
        test_prewarm_package_override,
        test_prompt_code_workflow_assembly,
        test_xml_tool_call_parsing,
        test_loader_basic,
        test_eval_imports_project_support_module,
        test_eval_resolves_relative_benchmark_path,
        test_loader_deep_nested_tree,
    ]
    async_tests = [
        test_prompt_eval_contract_is_gated,
        test_interactive_eval_feedback_is_gated,
        test_env_end_to_end,
        test_env_preserves_input_subdir,
        test_env_missing_output,
        test_loader_with_env,
    ]
    for t in sync_tests:
        try:
            name = t()
            print(f"  PASS  {name}")
            results.append((name, True))
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            results.append((t.__name__, False))
            failures += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            results.append((t.__name__, False))
            failures += 1

    for t in async_tests:
        try:
            name = await t()
            print(f"  PASS  {name}")
            results.append((name, True))
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            results.append((t.__name__, False))
            failures += 1
        except Exception as e:
            import traceback
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((t.__name__, False))
            failures += 1

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print()
    print(f"=== {passed}/{total} pass ===")
    return failures


if __name__ == '__main__':
    # Repo root on sys.path so `from envs.* import ...` works
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)
    failures = asyncio.run(main())
    sys.exit(0 if failures == 0 else 1)
