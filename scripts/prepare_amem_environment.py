"""Add missing A-MEM dependencies in a scratch venv, never in the agent conda env."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SENTENCE_TRANSFORMERS = "sentence-transformers==5.3.0"
PYTORCH_ENV = {"USE_TF": "0", "USE_TORCH": "1", "USE_FLAX": "0", "FORCE_TF_AVAILABLE": "0"}


def run(args):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    # Transformers reads these once at import; set them before Python starts.
    subprocess.run(args, check=True, env={**os.environ, **PYTORCH_ENV}, **options)


def constraints():
    # First distribution wins, matching the current sys.path precedence.
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            continue
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key != "sentence-transformers":
            versions.setdefault(key, dist.version)
    return "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))


def prepare(root):
    if importlib.util.find_spec("sentence_transformers") is not None:
        python = Path(sys.executable)
    else:
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        # A unique directory avoids concurrent installs from separate allocations.
        target = Path(tempfile.mkdtemp(prefix="amem-st530-", dir=root))
        constraint_file = target / "base-constraints.txt"
        constraint_file.write_text(constraints(), encoding="utf-8")
        run([sys.executable, "-m", "venv", "--system-site-packages", str(target)])
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(python), "-m", "pip", "--isolated", "install", "--only-binary=:all:",
             "--constraint", str(constraint_file), SENTENCE_TRANSFORMERS])
    # Import errors in an existing installation must not be mistaken for absence.
    run([str(python), "-c", "from sentence_transformers import SentenceTransformer; import sentence_transformers; print('AMEM_DEPENDENCIES_OK', sentence_transformers.__version__)"])
    return str(python.absolute())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python-file", type=Path, required=True)
    args = parser.parse_args()
    python = prepare(args.root)
    args.python_file.parent.mkdir(parents=True, exist_ok=True)
    args.python_file.write_text(python + "\n", encoding="utf-8")
    print(json.dumps({"event": "AMEM_AGENT_ENVIRONMENT", "python": python,
                      "base_python": sys.executable}))


if __name__ == "__main__":
    main()
