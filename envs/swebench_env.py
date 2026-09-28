"""SWE-bench patch-generation sandbox; grading happens in the official harness.

Only public task fields enter this module. Never mount the dataset, host home,
Docker socket, model credentials, gold patch, or test patch into the container.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import json
import re
import shlex
from uuid import uuid4

from agents.agent_text import extract_fn_call

_environments = ContextVar("swe_environments", default=None)
PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")


def public_instance(row):
    result = {key: row[key] for key in PUBLIC_FIELDS}
    if not all(isinstance(value, str) and value.strip() for value in result.values()):
        raise ValueError("SWE task fields must be nonempty strings")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+", result["instance_id"]):
        raise ValueError("Invalid SWE instance_id")
    if not re.fullmatch(r"[0-9a-f]{40}", result["base_commit"]):
        raise ValueError("Expected a full base commit SHA")
    return result


def image_name(instance_id):
    # Official SWE-bench v3.0.17 remote-image naming convention.
    return "swebench/sweb.eval.x86_64." + instance_id.lower().replace("__", "_1776_") + ":latest"


def repository_reset_command(base):
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Expected a full base commit SHA")
    # Only used inside the freshly created container's /testbed.
    return (
        f"set -euo pipefail; git checkout --detach {base} && git reset --hard {base} && git clean -ffd && "
        "git for-each-ref --format='delete %(refname)' | git update-ref --stdin && "
        "git reflog expire --expire=now --all && git gc --prune=now && "
        f"test \"$(git rev-parse HEAD)\" = {base} && test -z \"$(git status --porcelain)\""
    )


@contextmanager
def capture_environments():
    """Per-coroutine ownership lets the runner clean up even if an agent raises."""
    owned = []
    token = _environments.set(owned)
    try:
        yield owned
    finally:
        _environments.reset(token)


async def blocking_call(function, *args):
    # Cancellation must not leave a background thread creating a container after
    # the runner has already tried to remove it.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


class DockerSandbox:
    def __init__(self, task, *, memory="8g", cpus=4, tool_timeout=90):
        self.task = public_instance(task)
        self.memory, self.cpus, self.tool_timeout = memory, cpus, tool_timeout
        self.client = self.container = None
        self.provenance = {}

    def start(self):
        import docker
        self.client = docker.from_env(timeout=120)
        info = self.client.info()
        if info.get("OSType") != "linux" or info.get("Architecture") not in ("x86_64", "amd64"):
            raise RuntimeError("SWE runner requires a Linux x86_64 Docker daemon; Vista ARM is not supported")
        name = image_name(self.task["instance_id"])
        try:
            image = self.client.images.get(name)
        except docker.errors.ImageNotFound:
            image = self.client.images.pull(name, platform="linux/amd64")
        self.provenance = {"image": name, "image_id": image.id,
                           "repo_digests": image.attrs.get("RepoDigests", [])}
        self.container = self.client.containers.create(
            image.id, command=["sleep", "infinity"], entrypoint=[],
            name=f"ctxgraph-swe-{uuid4().hex}", working_dir="/testbed",
            network_disabled=True, cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"], pids_limit=512,
            mem_limit=self.memory, nano_cpus=int(self.cpus * 1e9),
            labels={"contextgraph.swe": "generation", "instance_id": self.task["instance_id"]},
        )
        self.container.start()
        base = self.task["base_commit"]
        status, output = self._exec(
            # Evaluation images can retain later repository refs/objects. Keep
            # only history reachable from the issue's base, so the agent cannot
            # inspect a future fix via tags, remote refs, reflogs or fsck.
            repository_reset_command(base) + " && "
            "test ! -e /eval.sh && test ! -e /patch.diff && test ! -e /tmp/patch.diff && "
            "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && python --version"
        )
        if status:
            raise RuntimeError(f"Unclean/misconfigured generation image: {output}")

    def _exec(self, command, *, limit=24000):
        if self.container is None:
            raise RuntimeError("Sandbox has not started")
        # GNU timeout bounds the entire process group, including child tests.
        api = self.client.api
        execution = api.exec_create(self.container.id,
            ["timeout", "--signal=TERM", "--kill-after=5", str(self.tool_timeout),
             "/bin/bash", "-c", command], workdir="/testbed")
        output, size, truncated = [], 0, False
        for chunk in api.exec_start(execution["Id"], stream=True):
            remaining = max(0, limit - size)
            if remaining:
                output.append(chunk[:remaining])
            size += min(len(chunk), remaining)
            truncated |= len(chunk) > remaining
        status = api.exec_inspect(execution["Id"])["ExitCode"]
        if status is None:
            raise RuntimeError("Docker exec exited without an exit code")
        text = b"".join(output).decode("utf-8", errors="replace")
        if truncated:
            text += "\n[output truncated]"
        return status, text

    def execute(self, code):
        command = ("source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "
                   "python -c " + shlex.quote(code))
        return self._exec(command)

    def patch(self):
        # Includes staged, unstaged, new, deleted, and model-committed files,
        # always relative to the dataset base commit (not mutable HEAD).
        status, patch = self._exec(
            "git add -N -- . && git -c core.quotePath=false diff --no-ext-diff --binary "
            + self.task["base_commit"] + " --", limit=8 * 1024 * 1024)
        if status or patch.endswith("\n[output truncated]"):
            raise RuntimeError("Patch extraction failed or exceeded 8 MiB")
        return patch

    def close(self):
        try:
            if self.container is not None:
                self.container.remove(force=True)
                self.container = None
        finally:
            if self.client is not None:
                self.client.close()


class SWEVerifiedEnv:
    prompt_domain = "swebench"

    def __init__(self, config, tokenizer, ability):
        self.config, self.ability = config, ability
        self.stats = Counter()
        self.is_finish = self.env_fail = False
        self.model_patch = None
        self.sandbox = None
        self.close_error = None
        owned = _environments.get()
        if owned is None:
            raise RuntimeError("SWEVerifiedEnv requires the eval_swebench_verified runner lifecycle")
        owned.append(self)

    async def init_env(self, item):
        task = public_instance(json.loads(self.ability.split("@", 1)[1]))
        settings = self.config.plugin
        self.sandbox = DockerSandbox(task, memory=settings.swe_memory,
            cpus=settings.swe_cpus, tool_timeout=settings.swe_tool_timeout)
        await blocking_call(self.sandbox.start)
        # Set this only after successful initialization; legacy agent loops catch
        # init errors, so they must not continue with a half-initialized task.
        self.instance_info = task

    async def run_action(self, response):
        self.stats["action"] += 1
        call = extract_fn_call(response)
        if not call:
            return {"observation": "No function call detected. Use python_exec or finish."}
        name, args = call["function"], call["arguments"]
        if name == "finish":
            self.is_finish = True
            self.stats["finish"] += 1
            self.stats["is_finish"] = 1
            return {"action": "finish"}
        if name != "python_exec" or not args.get("code", "").strip():
            return {"observation": "Invalid environment tool. Use python_exec with a code argument."}
        if self.env_fail:
            return {"observation": "Container infrastructure failed; this rollout will be marked error."}
        self.stats["python_exec"] += 1
        try:
            status, output = await blocking_call(self.sandbox.execute, args["code"])
        except Exception:
            self.env_fail = True
            raise
        self.stats["command_timeouts"] += int(status in (124, 137))
        return {"observation": f"[python_exec exit_code={status}]\n{output}"}

    async def get_reward(self, item, messages, context):
        if self.env_fail:
            raise RuntimeError("Cannot export a valid rollout after container infrastructure failure")
        self.model_patch = await blocking_call(self.sandbox.patch)
        # Required legacy agent-loop interface, NOT a benchmark score. The runner
        # drops reward/graph-reward fields and never reports this as accuracy.
        return "Official SWE-bench grading pending", 0.0, {"grading_pending": True}

    def close(self):
        if self.sandbox is not None:
            try:
                self.sandbox.close()
                self.close_error = None
            except Exception as exc:
                self.close_error = repr(exc)
                raise
