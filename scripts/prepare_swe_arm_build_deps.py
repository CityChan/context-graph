"""Run only in a fresh trusted base-image setup sandbox, before any patch.

Use syntax compatible with the historical Python versions inside Lite images.
The caller records stdout and clones this prepared environment for each grade.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def build_requirements(project):
    config = Path(project) / "pyproject.toml"
    if not config.exists():
        return []
    try:
        import tomllib as toml
    except ImportError:
        try:
            from pip._vendor import tomli as toml
        except ImportError:
            # Needed only in older images whose pip does not vendor a TOML reader.
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli==1.2.3"])
            import tomli as toml
    data = toml.loads(config.read_text(encoding="utf8"))
    requirements = data.get("build-system", {}).get("requires", [])
    if not isinstance(requirements, list) or any(not isinstance(item, str) or not item or item.startswith("-") for item in requirements):
        raise ValueError("Invalid build-system.requires in base checkout")
    return requirements


def main():
    if sys.prefix != "/opt/miniconda3/envs/testbed":
        raise RuntimeError("Build preparation must use the private testbed environment")
    # Do not inherit offline pip policy from a previous grading invocation.
    for key in ("PIP_NO_INDEX", "PIP_NO_DEPS", "PIP_NO_BUILD_ISOLATION"):
        os.environ.pop(key, None)
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "30"
    os.environ["PIP_RETRIES"] = "2"
    requirements = build_requirements("/testbed")
    print("SWE_ARM_BUILD_REQUIREMENTS " + json.dumps(requirements), flush=True)
    print("SWE_ARM_PACKAGES_BEFORE", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "freeze", "--all"])
    if requirements:
        # Resolve every declared requirement (including exact Cython pins), rather
        # than patching one missing import at a time. No project/gold/model patch
        # is installed in this network-enabled phase.
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
    print("SWE_ARM_PACKAGES_AFTER", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "freeze", "--all"])
    print("SWE_ARM_BUILD_ENV_READY", flush=True)


if __name__ == "__main__":
    main()
