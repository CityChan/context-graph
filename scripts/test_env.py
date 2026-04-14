#!/usr/bin/env python3
"""Environment sanity check for FoldAgent.

Usage (login node — no GPU needed):
    conda activate foldagent
    cd /work/09281/chc_1996/vista/context-graph
    python scripts/test_env.py
"""

import sys
import importlib

PASS = "\033[32mOK\033[0m"
FAIL = "\033[31mMISSING\033[0m"
WARN = "\033[33mWARN\033[0m"

results = {"ok": [], "missing": [], "warn": []}


def check(name, required=True):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "")
        tag = f"{name} {ver}".strip()
        print(f"  [{PASS}] {tag}")
        results["ok"].append(name)
        return mod
    except Exception as e:
        if required:
            print(f"  [{FAIL}] {name}  ({e})")
            results["missing"].append(name)
        else:
            print(f"  [{WARN}] {name}  ({e})")
            results["warn"].append(name)
        return None


# ── 1. Core ML ──────────────────────────────────────────────
print("\n=== Core ML ===")
torch = check("torch")
if torch:
    if torch.cuda.is_available():
        print(f"         CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  [{WARN}] CUDA not available (ok on login node)")
        results["warn"].append("cuda")
check("transformers")
check("peft")
check("accelerate")
check("datasets")

# ── 2. Attention backend ───────────────────────────────────
print("\n=== Attention Backend ===")
fa = check("flash_attn", required=False)
if fa:
    print(f"         flash_attn available — will use flash_attention_2")
else:
    print(f"         flash_attn not installed — will fallback to eager (OK)")

# ── 3. Rollout Engine ──────────────────────────────────────
print("\n=== Rollout Engine ===")
vllm = check("vllm", required=False)
sglang = check("sglang", required=False)
if vllm is None and sglang is None:
    print(f"  [{FAIL}] Need at least one of vllm or sglang for rollout")
    results["missing"].append("vllm-or-sglang")

# ── 4. Distributed / Training ──────────────────────────────
print("\n=== Distributed / Training ===")
check("ray")
check("torchdata")
check("tensordict")
check("hydra")
check("omegaconf")
check("wandb", required=False)
check("tensorboard", required=False)

# ── 5. Web / API ───────────────────────────────────────────
print("\n=== Web / API ===")
check("fastapi")
check("uvicorn")
check("httpx")
check("aiohttp")
check("pydantic")
check("openai")

# ── 6. Eval / Math ─────────────────────────────────────────
print("\n=== Eval / Math ===")
check("math_verify")
check("latex2sympy2_extended")
check("pylatexenc")

# ── 7. Misc ────────────────────────────────────────────────
print("\n=== Misc ===")
check("unidiff")
check("psutil")
check("pandas")
check("numpy")
check("tqdm")
check("filelock")
check("packaging")
check("cloudpickle")
check("cachetools")
check("codetiming", required=False)
check("liger_kernel", required=False)
check("requests")

# ── 8. verl + FoldAgent internal imports ───────────────────
print("\n=== verl / FoldAgent ===")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

check("verl")
try:
    from verl import DataProto
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    print(f"  [{PASS}] verl internals (DataProto, AgentLoopBase, PPO)")
    results["ok"].append("verl-internals")
except Exception as e:
    print(f"  [{FAIL}] verl internals  ({e})")
    results["missing"].append("verl-internals")

try:
    from agents.tool_spec import search_tool, codeact_tool, branch_tool
    from agents.prompts import create_chat
    print(f"  [{PASS}] agents (tool_spec, prompts)")
    results["ok"].append("agents")
except Exception as e:
    print(f"  [{FAIL}] agents  ({e})")
    results["missing"].append("agents")

# ── 9. verl flash_attn fallback ────────────────────────────
print("\n=== verl attn fallback ===")
try:
    from verl.workers.fsdp_workers import _VERL_DEFAULT_ATTN
    print(f"  [{PASS}] verl default attn: {_VERL_DEFAULT_ATTN}")
    results["ok"].append("verl-attn-fallback")
except ImportError:
    print(f"  [{WARN}] verl _VERL_DEFAULT_ATTN not found (fsdp_workers not patched)")
    results["warn"].append("verl-attn-fallback")

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"OK: {len(results['ok'])}  |  WARN: {len(results['warn'])}  |  MISSING: {len(results['missing'])}")
if results["missing"]:
    print(f"\nMissing (install these):")
    for m in results["missing"]:
        print(f"  pip install {m}")
if results["warn"]:
    print(f"\nWarnings (optional):")
    for m in results["warn"]:
        print(f"  {m}")
print()
sys.exit(1 if results["missing"] else 0)
