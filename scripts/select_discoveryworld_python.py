"""Find a compatible existing agent Python without modifying any environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROBE = 'import sys; print("%d.%d" % sys.version_info[:2], flush=True)'


def check_python(path):
    """Check only the interpreter version; never initialize site or scan packages.

    -I alone still processes .pth and sitecustomize. On shared environments those
    startup hooks can stall, so version selection also needs -S. The normal agent
    dependency/simulator preflight remains separate and retains site initialization.
    """
    try:
        result = subprocess.run([str(path), "-I", "-S", "-c", PROBE], capture_output=True,
                                text=True, timeout=30, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        version = tuple(int(part) for part in result.stdout.strip().split("."))
        if len(version) != 2:
            raise ValueError("Expected major.minor version output")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Version probe timed out for {path}; compatibility is unknown "
                           "(site initialization and package scanning were disabled)") from exc
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Cannot inspect {path}: {exc}") from exc
    if version not in ((3, 10), (3, 11)):
        raise RuntimeError(f"{path}: Python {version[0]}.{version[1]}; DiscoveryWorld requires 3.10/3.11")
    return {"version": list(version)}


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
    raise RuntimeError("Could not confirm a compatible agent Python. Set BENCH_BASE_PYTHON to an existing "
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
