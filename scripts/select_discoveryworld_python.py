"""Find a compatible existing agent Python without modifying any environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROBE = ('import sys,json,importlib.util; '
         'print(json.dumps({"version":list(sys.version_info[:2]),'
         '"missing":[m for m in ("torch","numpy","transformers","omegaconf","tensordict") '
         'if importlib.util.find_spec(m) is None]}))')


def check_python(path):
    """Check version and inherited agent packages before creating an overlay."""
    try:
        result = subprocess.run([str(path), "-I", "-c", PROBE], capture_output=True,
                                text=True, timeout=30, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        info = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Cannot inspect {path}: {exc}") from exc
    version = tuple(info["version"])
    if version not in ((3, 10), (3, 11)):
        raise RuntimeError(f"{path}: Python {version[0]}.{version[1]}; DiscoveryWorld requires 3.10/3.11")
    if info["missing"]:
        raise RuntimeError(f"{path}: missing agent packages: {', '.join(info['missing'])}")
    return info


def candidates(environ):
    paths = ["/work/09281/chc_1996/vista/miniconda3/envs/cxtgraph/bin/python"]
    if environ.get("CONDA_EXE"):
        paths.append(str(Path(environ["CONDA_EXE"]).parent.parent / "envs/cxtgraph/bin/python"))
    if environ.get("CONDA_PREFIX"):
        paths.append(str(Path(environ["CONDA_PREFIX"]) / "bin/python"))
    paths.append("/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP/bin/python")
    paths.extend(filter(None, (shutil.which(name) for name in ("python3.11", "python3.10", "python"))))
    return list(dict.fromkeys(paths))


def select_python(environ=None):
    environ = os.environ if environ is None else environ
    explicit = environ.get("BENCH_BASE_PYTHON")
    if explicit:
        check_python(explicit)  # An explicit incompatible override must not be silently ignored.
        return os.path.abspath(explicit)
    failures = []
    for path in candidates(environ):
        if not Path(path).is_file():
            continue
        try:
            check_python(path)
            # Do not resolve symlinks: a venv's bin/python must retain its prefix.
            return os.path.abspath(path)
        except RuntimeError as exc:
            failures.append(str(exc))
    raise RuntimeError("No compatible agent Python found. Set BENCH_BASE_PYTHON to an existing "
                       "Python 3.10/3.11 agent environment. Checked: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", help="Validate one interpreter without choosing a fallback")
    args = parser.parse_args()
    try:
        if args.check:
            info = check_python(args.check)
            print(json.dumps({"python": args.check, **info}))
        else:
            print(select_python())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
