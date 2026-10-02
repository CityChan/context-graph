"""Pinned ARM sandbox: default pilot or explicit subset image registry; not official x86."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

from envs.swebench_env import BoundedOutput, public_instance, repository_reset_command

INSTANCE = "sympy__sympy-20590"
BASE = "cffd4e0f86fefd4802349a9f9b19ed70934ea354"
DIGEST = "a8b2a5265717391b168a5d7aa884b466e80fa748d074373a56d328cc48d887b9"
IMAGE_NAME = f"{INSTANCE}-arm64-{DIGEST}.sif"


def image_record(root, task):
    task = public_instance(task)
    registry = Path(root) / "arm-images.json"
    if registry.exists():
        records = json.loads(registry.read_text(encoding="utf8"))
        record = records[task["instance_id"]]
        if record["base_commit"] != task["base_commit"] or record["repo"] != task["repo"]:
            raise ValueError("ARM image task identity mismatch")
        digest = record["oci_digest"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Expected pinned ARM image digest")
        return f'{task["instance_id"]}-arm64-{digest}.sif', digest
    if task["instance_id"] != INSTANCE or task["base_commit"] != BASE or task["repo"] != "sympy/sympy":
        raise ValueError("ARM pilot supports only the pinned sympy__sympy-20590 instance")
    return IMAGE_NAME, DIGEST


def checked_image(root, task):
    name, _ = image_record(root, task)
    image = Path(root).resolve() / "images" / name
    checksum = Path(str(image) + ".sha256").read_text().split()
    if len(checksum) != 2 or checksum[1].lstrip("*") != name:
        raise ValueError("Invalid image checksum record; run the Apptainer preflight")
    digest = hashlib.sha256()
    with image.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != checksum[0]:
        raise ValueError("Cached ARM image checksum mismatch")
    return image, digest.hexdigest()


class ApptainerSandbox:
    def __init__(self, task, *, root, memory="8g", cpus=1, tool_timeout=90):
        self.task = public_instance(task)
        self.root = Path(root).resolve()
        self.tool_timeout = tool_timeout
        self.work = None
        self.runtime = None
        self.runtime_ready = False
        self.provenance = {}

    def start(self):
        if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
            raise RuntimeError("ARM Apptainer pilot requires a Linux ARM compute node")
        if platform.node().split(".")[0].startswith("login"):
            raise RuntimeError("Apptainer cannot run on TACC login nodes")
        self.image, sha = checked_image(self.root, self.task)
        work_root = self.root / "sandboxes"
        work_root.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="sympy-", dir=work_root))
        self.provenance = {"backend": "apptainer-arm-pilot", "image": str(self.image),
                           "sif_sha256": sha, "oci_digest": image_record(self.root, self.task)[1],
                           "resource_limits": "allocation limits; one numerical thread; no per-task RAM cgroup"}
        # Only this fixed initialization command sees the original image /testbed.
        status, output = self._exec(
            "set -e; test ! -e /eval.sh; test ! -e /patch.diff; "
            "test ! -e /tmp/patch.diff; cp -a --no-preserve=ownership /testbed/. /workspace/; "
            "cd /workspace; " + repository_reset_command(self.task["base_commit"]).replace("git gc", "git -c pack.threads=1 gc"),
            initializing=True, timeout=600)
        if status:
            raise RuntimeError(f"ARM sandbox initialization failed: {output}")

    def prepare_grading_environment(self, template=None):
        """Own a fresh writable copy at the original prefix; never mutate the SIF."""
        if self.work is None or self.runtime is not None:
            raise RuntimeError("Expected a started sandbox without a grading environment")
        self.runtime = Path(tempfile.mkdtemp(prefix="runtime-", dir=self.root / "sandboxes"))
        if template is not None:
            # Each grade owns its copy; no package install can mutate calibration's
            # prepared template or another task's runtime. Preserve conda symlinks.
            shutil.copytree(template, self.runtime, symlinks=True, dirs_exist_ok=True)
            self.runtime_ready = True
            self.provenance["grading_environment"] = "private copy of prepared build environment"
            return
        status, output = self._exec(
            "set -e; cp -a --no-preserve=ownership /opt/miniconda3/envs/testbed/. /runtime-copy/; "
            "chmod -R u+rwX /runtime-copy", timeout=600)
        if status:
            raise RuntimeError(f"Cannot prepare writable grading environment: {output}")
        self.runtime_ready = True
        self.provenance["grading_environment"] = "private writable copy at original testbed prefix"

    def _exec(self, command, *, limit=24000, initializing=False, timeout=None, inputs=None,
              stdout_only=False, preserve_tail=False, trusted_setup_network=False):
        if self.work is None:
            raise RuntimeError("Sandbox has not started")
        target = "/workspace" if initializing else "/testbed"
        argv = ["apptainer", "exec", "--cleanenv", "--containall", "--no-home",
                "--no-mount", "hostfs,bind-paths,cwd",
                "--bind", f"{self.work}:{target}", "--pwd", target]
        if not trusted_setup_network:
            argv += ["--net", "--network", "none"]
        if inputs is not None:
            argv += ["--bind", f"{Path(inputs).resolve()}:/grading-input:ro"]
        if self.runtime is not None:
            runtime_target = "/opt/miniconda3/envs/testbed" if self.runtime_ready else "/runtime-copy"
            argv += ["--bind", f"{self.runtime}:{runtime_target}"]
        argv += [str(self.image), "/bin/bash", "--noprofile", "--norc", "-c",
                 "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1; " + command]
        # Strip host credentials, preload libraries, and implicit container binds.
        env = {key: os.environ[key] for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG") if key in os.environ}
        env["APPTAINER_CACHEDIR"] = str(self.root / "apptainer-cache")
        env["GOMAXPROCS"] = "1"
        # GNU timeout supervises the Apptainer process group, including children.
        argv = ["timeout", "--signal=TERM", "--kill-after=5", str(timeout or self.tool_timeout)] + argv
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as diagnostics:
            result = subprocess.run(argv, env=env, cwd=self.work, stdout=output,
                                    stderr=diagnostics if stdout_only else subprocess.STDOUT)
            output.seek(0)
            content = BoundedOutput(limit, preserve_tail)
            for block in iter(lambda: output.read(65536), b""):
                content.append(block)
            if stdout_only and result.returncode:
                diagnostics.seek(0)
                detail = diagnostics.read(limit).decode("utf-8", errors="replace")
                raise RuntimeError(f"ARM patch extraction failed (exit {result.returncode}): {detail}")
        return result.returncode, content.text()

    def execute(self, code):
        return self._exec("source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && python -c " + shlex.quote(code),
                          preserve_tail=True)

    def patch(self):
        status, patch = self._exec("git add -N -- . && git -c core.quotePath=false diff --no-ext-diff --binary " + self.task["base_commit"] + " --", limit=8 * 1024 * 1024, stdout_only=True)
        if status or patch.endswith("\n[output truncated]"):
            raise RuntimeError("ARM patch extraction failed or exceeded 8 MiB")
        return patch

    def close(self):
        if self.runtime is not None:
            runtime = self.runtime.resolve()
            if runtime.parent != (self.root / "sandboxes").resolve() or not runtime.name.startswith("runtime-"):
                raise RuntimeError("Refusing to clean an unmanaged grading environment")
            shutil.rmtree(runtime)
            self.runtime = None
            self.runtime_ready = False
        if self.work is not None:
            # Delete only the fresh directory created by this sandbox instance.
            work = self.work.resolve()
            if work.parent != (self.root / "sandboxes").resolve() or not work.name.startswith("sympy-"):
                raise RuntimeError("Refusing to clean a sandbox outside its managed root")
            shutil.rmtree(work)
            self.work = None
