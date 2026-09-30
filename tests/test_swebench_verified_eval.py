import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.prompts_code import create_chat_code
from envs.swebench_env import (
    DockerSandbox, SWEVerifiedEnv, capture_environments, image_name, public_instance, repository_reset_command,
)
from scripts.eval_swebench_verified import (
    config_for, file_hash, grading_summary, select_tasks, validate_predictions, write_json,
)


def task(number=1):
    return {"instance_id": f"django__django-{number}", "repo": "django/django",
            "base_commit": "a" * 40, "problem_statement": "Fix the reported bug"}


def args(method="contextgraph"):
    return SimpleNamespace(method=method, max_turn=100, task_timeout=3600, memory="8g", cpus=4)


def test_public_task_boundary_excludes_gold_and_hints():
    row = dict(task(), patch="GOLD", test_patch="SECRET_TEST", hints_text="SECRET_HINT",
               FAIL_TO_PASS=["hidden_test"], PASS_TO_PASS=["hidden_regression"])
    assert public_instance(row) == task()
    for method, workflow in (("react", "code"), ("foldagent", "code_branch"), ("contextgraph", "code_graph")):
        chat = create_chat_code(public_instance(row)["problem_statement"], workflow,
                                env=SimpleNamespace(prompt_domain="swebench"), expose_graph_tools=False)
        text = json.dumps(chat)
        assert not any(secret in text for secret in ("GOLD", "SECRET_TEST", "SECRET_HINT", "hidden_test", "pred_results"))
        assert "Files persist; Python variables do not" in text
        assert ("same repository" in text) == (method != "react")


@pytest.mark.parametrize("key,value", [("instance_id", "../../bad"), ("base_commit", "HEAD; echo bad"), ("repo", "")])
def test_reject_malformed_task(key, value):
    with pytest.raises(ValueError):
        public_instance(dict(task(), **{key: value}))


def test_deterministic_paired_selection_and_unknown_id_rejected():
    rows = [task(i) for i in range(20)]
    assert select_tasks(rows, 5, 42) == select_tasks(rows[::-1], 5, 42)
    assert len(select_tasks(rows, -1, 42)) == 20
    with pytest.raises(ValueError):
        select_tasks(rows, 5, 42, ["missing"])
    with pytest.raises(ValueError):
        select_tasks(rows + rows, -1, 42)


@pytest.mark.parametrize("context_length", [32768, 65536])
def test_matched_budgets_and_code_workflows(context_length):
    method_args = [args(method) for method in ("react", "foldagent", "contextgraph")]
    for method_arg in method_args:
        method_arg.context_length = context_length
    configs = [config_for(method_arg).actor_rollout_ref.rollout for method_arg in method_args]
    assert [c.plugin.workflow for c in configs] == ["code", "code_branch", "code_graph"]
    assert all(c.prompt_length == 8192 and c.response_length == context_length - 8192 for c in configs)
    assert all(c.plugin.branch_len == context_length - 8192 for c in configs)
    assert all(c.plugin.val_response_length == context_length - 8192 for c in configs)
    assert all(c.plugin.turn_max_new_tokens == 2048 for c in configs)
    assert all(c.plugin.process_reward is None for c in configs)


class FakeSandbox:
    instances = []

    def __init__(self, task, **kwargs):
        self.closed = False
        self.code = []
        self.provenance = {"image_id": "sha256:fake"}
        self.instances.append(self)

    def start(self):
        pass

    def execute(self, code):
        self.code.append(code)
        return 0, "test output"

    def patch(self):
        return "diff --git a/x b/x\n"

    def close(self):
        self.closed = True


def test_environment_executes_only_in_backend_and_returns_pending_grade(monkeypatch):
    monkeypatch.setattr("envs.swebench_env.DockerSandbox", FakeSandbox)
    async def run():
        with capture_environments() as owned:
            env = SWEVerifiedEnv(config_for(args()).actor_rollout_ref.rollout, None,
                                 "SWEVerified@" + json.dumps(task()))
            await env.init_env(None)
            response = await env.run_action("<function=python_exec><parameter=code>print(1)</parameter></function>")
            assert "test output" in response["observation"]
            assert env.sandbox.code == ["print(1)"]
            await env.run_action("<function=finish><parameter=message>done</parameter></function>")
            assert env.is_finish
            message, _, metadata = await env.get_reward(None, None, None)
            assert metadata == {"grading_pending": True}
            assert "pending" in message and env.model_patch
            assert owned == [env]
            env.close()
            assert env.sandbox.closed
    asyncio.run(run())


def test_failed_initialization_remains_owned_for_cleanup(monkeypatch):
    class Broken(FakeSandbox):
        def start(self):
            raise RuntimeError("container setup failed")
    monkeypatch.setattr("envs.swebench_env.DockerSandbox", Broken)
    async def run():
        with capture_environments() as owned:
            env = SWEVerifiedEnv(config_for(args()).actor_rollout_ref.rollout, None,
                                 "SWEVerified@" + json.dumps(task()))
            with pytest.raises(RuntimeError):
                await env.init_env(None)
            assert not hasattr(env, "instance_info")
            owned[0].close()
            assert env.sandbox.closed
    asyncio.run(run())


def test_docker_launch_has_no_host_mount_network_or_privileged_mode(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    client = MagicMock()
    client.info.return_value = {"OSType": "linux", "Architecture": "x86_64"}
    image = client.images.get.return_value
    image.id, image.attrs = "sha256:fixed", {"RepoDigests": ["image@sha256:fixed"]}
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(from_env=lambda **kwargs: client))
    sandbox = DockerSandbox(task())
    monkeypatch.setattr(sandbox, "_exec", lambda *a, **kw: (0, "Python 3.9"))
    sandbox.start()
    positional, settings = client.containers.create.call_args
    assert positional == ("sha256:fixed",)
    assert settings["network_disabled"] is True
    assert settings["cap_drop"] == ["ALL"]
    assert "volumes" not in settings and "privileged" not in settings and "environment" not in settings
    assert sandbox.provenance["image_id"] == "sha256:fixed"
    assert image_name("django__django-1") == "swebench/sweb.eval.x86_64.django_1776_django-1:latest"
    sandbox.close()
    client.containers.create.return_value.remove.assert_called_once_with(force=True)


def test_container_execution_uses_timeout_and_caps_output():
    from unittest.mock import MagicMock
    sandbox = DockerSandbox(task())
    sandbox.client = MagicMock()
    sandbox.container = SimpleNamespace(id="container")
    api = sandbox.client.api
    api.exec_create.return_value = {"Id": "exec"}
    api.exec_start.return_value = iter([b"abcdef", b"more"])
    api.exec_inspect.return_value = {"ExitCode": 124}
    status, text = sandbox._exec("print test", limit=3)
    assert status == 124 and text == "abc\n[output truncated]"
    assert api.exec_create.call_args.args[1][:4] == ["timeout", "--signal=TERM", "--kill-after=5", "90"]


def test_patch_extraction_rejects_truncation(monkeypatch):
    sandbox = DockerSandbox(task())
    monkeypatch.setattr(sandbox, "_exec", lambda *a, **kw: (0, "diff\n[output truncated]"))
    with pytest.raises(RuntimeError):
        sandbox.patch()


def test_git_patch_includes_committed_staged_unstaged_new_and_deleted_files(tmp_path, monkeypatch):
    import os
    import shutil
    import subprocess
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.exists() else shutil.which("bash")
    if not bash or not shutil.which("git"):
        pytest.skip("Git and Bash are required for the real patch-export fixture")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=tmp_path, text=True,
                                       creationflags=flags, stderr=subprocess.DEVNULL).strip()
    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("config", "core.autocrlf", "false")
    for name in ("committed.py", "staged.py", "unstaged.py", "deleted.py"):
        (tmp_path / name).write_text("before\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "future-gold.txt").write_text("future solution\n")
    git("add", ".")
    git("commit", "-m", "future fix")
    future = git("rev-parse", "HEAD")
    git("tag", "future-tag")
    git("update-ref", "refs/remotes/origin/future", future)
    reset = subprocess.run([bash, "-c", repository_reset_command(base)], cwd=tmp_path,
                           capture_output=True, text=True, creationflags=flags, timeout=30)
    assert reset.returncode == 0, reset.stderr
    assert not (tmp_path / "future-gold.txt").exists()
    assert git("for-each-ref") == ""
    with pytest.raises(subprocess.CalledProcessError):
        git("cat-file", "-e", future)
    (tmp_path / "committed.py").write_text("after\n")
    git("add", "committed.py")
    git("commit", "-m", "agent commit")
    (tmp_path / "staged.py").write_text("after\n")
    git("add", "staged.py")
    (tmp_path / "unstaged.py").write_text("after\n")
    (tmp_path / "new.py").write_text("new\n")
    (tmp_path / "deleted.py").unlink()
    sandbox = DockerSandbox(dict(task(), base_commit=base))
    def execute_internal_git(command, **kwargs):
        # Only the adapter's own fixed git-export command runs on this fixture;
        # no model-generated code ever executes on the host.
        result = subprocess.run([bash, "-c", command], cwd=tmp_path, capture_output=True,
                                text=True, creationflags=flags, timeout=30)
        return result.returncode, result.stdout
    monkeypatch.setattr(sandbox, "_exec", execute_internal_git)
    patch = sandbox.patch()
    assert all(name in patch for name in ("committed.py", "staged.py", "unstaged.py", "new.py", "deleted.py"))
    assert "new file mode" in patch and "deleted file mode" in patch


def grading_fixture(root):
    predictions = [{"instance_id": task(i)["instance_id"], "model_name_or_path": "react--Qwen/9B",
                    "model_patch": "diff" if i != 3 else ""} for i in (1, 2, 3)]
    write_json(root / "manifest.json", {"method": "react", "instance_ids": [p["instance_id"] for p in predictions]})
    for name, rows in (("predictions.jsonl", predictions), ("results.jsonl", [
            {"instance_id": p["instance_id"], "status": "generated"} for p in predictions])):
        (root / name).write_text("\n".join(json.dumps(p) for p in rows), encoding="utf-8")
    write_json(root / "generation_summary.json", {"predictions_sha256": file_hash(root / "predictions.jsonl")})
    grading = root / "grading"
    grading.mkdir()
    write_json(grading / "grading_manifest.json", {"predictions_sha256": file_hash(root / "predictions.jsonl")})
    for identity, resolved in (("django__django-1", True), ("django__django-2", False)):
        report_dir = grading / "logs/run_evaluation/official/react--Qwen__9B" / identity
        report_dir.mkdir(parents=True)
        write_json(report_dir / "report.json", {identity: {"resolved": resolved}})
    return grading


def test_official_reports_only_and_empty_patches_remain_in_denominator(tmp_path):
    grading = grading_fixture(tmp_path)
    summary = grading_summary(tmp_path, grading)
    assert summary["pass_at_1"] == pytest.approx(1 / 3)
    assert summary["resolved"] == summary["unresolved"] == summary["empty_patches"] == 1


def test_missing_report_is_infrastructure_error_not_wrong_answer(tmp_path):
    grading = grading_fixture(tmp_path)
    (grading / "logs/run_evaluation/official/react--Qwen__9B/django__django-2/report.json").unlink()
    with pytest.raises(RuntimeError, match="missing/invalid"):
        grading_summary(tmp_path, grading)
    summary = json.loads((grading / "summary.json").read_text())
    assert summary["pass_at_1"] is None and summary["harness_errors"] == 1


def test_changed_predictions_and_incomplete_runs_rejected(tmp_path):
    grading_fixture(tmp_path)
    with (tmp_path / "predictions.jsonl").open("a") as handle:
        handle.write("\n" + json.dumps({"instance_id": "extra", "model_patch": ""}))
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        validate_predictions(tmp_path)
