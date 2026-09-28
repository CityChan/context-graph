# SWE-bench Verified evaluation

This adapter generates repository patches with ReAct, FoldAgent, or isolated
ContextGraph, then evaluates those patches with the official SWE-bench harness.
It does **not** use the BC-P search corpus, answer matching, or an LLM judge.
No training is performed. Default model: `Qwen/Qwen3.5-9B`.

## Execution layout

- **Model server:** one Vista GPU node, or an existing compatible vLLM server.
- **Agent runner and repository containers:** a machine with access to a Linux
  **x86_64 Docker daemon**. CPU-only is sufficient here; inference is remote.
- **Official grader:** a separate Python environment on the same Docker host.
  Keep its instance images until grading is finished.

Vista ARM nodes are not supported as repository-container hosts by this initial
adapter. Do not submit the generation/grading script there expecting `sbatch`
alone to provide Docker. Docker Desktop with Linux amd64 containers can be used
on an x86 Windows machine; Linux x86_64 is the preferred runner. No Modal backend
is implemented. Running a cloud service would be a separate setup decision.

The default is a **5-task smoke evaluation**, not a 500-task benchmark score.
Each selected task gets one attempt. `--samples -1` selects all 500 Verified test
instances. Selection is deterministic from sorted instance IDs and seed 42.
Use identical data, selection, model revision and budgets for all methods.

## Data and protocol

`prepare` downloads `princeton-nlp/SWE-bench_Verified`, resolves its revision to
an immutable HF commit, requires 500 unique test instances, and writes:

- `public/instances.json`: instance ID, repository, base commit, issue text only.
- `grading/instances.json`: original dataset, including gold tests, for the grader.
- `manifest.json`: revision, counts and SHA-256 hashes of both files.

Only the public fields are given to the agent. The dataset directory, gold
patch, test patch, hints, model credentials and host filesystem are never
mounted into generation containers. Each task starts a fresh official instance
image, resets `/testbed` to its base commit, checks it is clean, disables network,
removes later refs/reflogs/unreachable Git objects, and applies CPU/memory/process
limits. History reachable from the base commit remains available. Generation records the image ID/digest.
Grading rejects a locally changed image tag. A remote Docker daemon must be
trusted; its administrator can see container contents.

Both branch methods use the **same delegation instruction**. All three methods
use `python_exec` (fresh Python process, persistent files), with `pathlib` for
edits and `subprocess` for shell/test commands. Tool processes time out after
90 seconds; observations are capped at 24,000 bytes. `finish` submits the Git
diff, including new/staged/deleted files and changes already committed by the
model. Oversized/truncated patch export is an error, not an empty success.

ReAct/FoldAgent reuse `agents/fold_agent_code.py` with `code`/`code_branch`;
ContextGraph reuses `agents/graph_agent_code_isolated.py` with `code_graph` and
the structured graph controller. Branches isolate conversational memory while
sharing the same repository, and run sequentially. This is our **code-domain
agent evaluation**, not a reproduction of an upstream published SWE score or
the identical search-domain `memory=repaired` BC-P configuration.

Defaults: 32,768 context (8,192 prompt + 24,576 response allowance), 2,048 output
tokens per call, 100 main-loop iterations, up to 10 branches, 90-second command
timeout, 3,600-second task timeout, 8 GiB / 4 CPUs per container, one worker.
Branch/controller calls also consume time/tokens; the context allowance is not
a fixed total generation-token cap over all branches. Raise worker count only
when Docker-host resources and the vLLM server can sustain it.

## Setup on the Docker host

From a checkout of this repository, create a CPU agent environment (Bash/Linux
commands below). The local `verl` package is imported directly from the checkout.
No local GPU/vLLM package is needed. Keep official-grader dependencies separate:

```bash
python -m venv .venv-swe-agent
.venv-swe-agent/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-swe-agent/bin/python -m pip install -r requirements_swebench_generation.txt
python -m venv .venv-swe-grade
.venv-swe-grade/bin/python -m pip install -r requirements_swebench_eval.txt
source .venv-swe-agent/bin/activate
docker info
python scripts/eval_swebench_verified.py prepare --data-dir data/swebench-verified
```

On Windows the venv executables are under `Scripts` instead of `bin`; run the
Python CLI commands in the corresponding environment. The shell launch commands
later in this document are for Vista Bash or Linux, as labeled.

`requirements_swebench_eval.txt` pins official SWE-bench **v3.0.17**, commit
`3f01bd622c0a22c00406139f69a234ef08225f22`. The adapter uses that version's
image naming, CLI and report schema; it deliberately does not track upstream
`main`. The grader checks the installed VCS commit before executing.

If this host has no tokenizer copy, download only small tokenizer/config files
from the same checkpoint revision as the Vista model:

```bash
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --include '*.json' '*.jinja' '*.txt' --local-dir cache/qwen35-9b-tokenizer
```

## Optional Vista model server

On the Vista login node, from the project root:

```bash
mkdir -p logs
sbatch scripts/serve_swe_qwen35_9b_vista.sbatch
```

This requests **one gh node for one hour**, starts only the model server, resolves
the existing 9B checkpoint on `$SCRATCH` offline, and reuses the tested Vista
CUDA/compiler/TLS/cache setup from the BC-P launcher. No checkpoint download or
search server is started. The Slurm log prints the allocated hostname/port.
One hour is an initial server allocation, **not** an estimate for all 500 tasks.

From the Docker host, open a tunnel through your configured Vista SSH alias,
replacing `COMPUTE_HOST` with that hostname:

```bash
ssh -N -L 18000:COMPUTE_HOST:18000 vista
```

Keep the tunnel running. Use `--endpoint http://127.0.0.1:18000`, without `/v1`.
The runner preflights token-ID output and, for ContextGraph, structured JSON
output. The endpoint and tokenizer must correspond to the same weights/template.

## Generate patches

From the Docker host's agent environment, with the server and tunnel ready:

```bash
python -u scripts/eval_swebench_verified.py generate --method react --data-dir data/swebench-verified --model-path cache/qwen35-9b-tokenizer --endpoint http://127.0.0.1:18000 --output outputs/swe-9b-react-smoke
python -u scripts/eval_swebench_verified.py generate --method foldagent --data-dir data/swebench-verified --model-path cache/qwen35-9b-tokenizer --endpoint http://127.0.0.1:18000 --output outputs/swe-9b-foldagent-smoke
python -u scripts/eval_swebench_verified.py generate --method contextgraph --data-dir data/swebench-verified --model-path cache/qwen35-9b-tokenizer --endpoint http://127.0.0.1:18000 --output outputs/swe-9b-contextgraph-smoke
```

Run the methods sequentially unless you provisioned enough inference capacity.
For a chosen task use, for example, `--instance-ids django__django-10097`.
For the full split add `--samples -1` and choose a new output directory.

Files stream per task into `instances/INSTANCE_ID/`: `requests.jsonl`,
`trajectory.json`, `model.patch`, `prediction.json`, `result.json`, and on
failure `error.txt`. Aggregate `results.jsonl` and official-format
`predictions.jsonl` are appended immediately after each task. The latter has
`instance_id`, `model_name_or_path`, and `model_patch`.

`generation_summary.json` reports generated/error/empty-patch counts and says
`grading_status: pending`. **It is not an accuracy report.** Legacy code-loop
reward placeholders are stripped from exported stats. A finished conversation
or a nonempty patch does not mean the issue was resolved.

Existing output directories are rejected, to prevent accidental result mixing.
An interrupted run preserves completed artifacts but cannot be graded as a full
run; there is no automatic resume in this first adapter. Inspect failed tasks
and rerun a clearly identified subset into a fresh directory. Do not silently
drop failures or average only successfully generated instances.

## Grade using the official harness

On the same Docker host (substitute each method's output directory):

```bash
.venv-swe-grade/bin/python scripts/eval_swebench_verified.py grade --data-dir data/swebench-verified --output outputs/swe-9b-react-smoke
.venv-swe-grade/bin/python scripts/eval_swebench_verified.py grade --data-dir data/swebench-verified --output outputs/swe-9b-foldagent-smoke
.venv-swe-grade/bin/python scripts/eval_swebench_verified.py grade --data-dir data/swebench-verified --output outputs/swe-9b-contextgraph-smoke
```

The wrapper invokes `python -m swebench.harness.run_evaluation` with the exact
pinned dataset subset, saved predictions, official `swebench` image namespace,
and a fresh grading working directory. See `grading/harness.log` and
`grading/logs/run_evaluation/official/MODEL/INSTANCE_ID/` for test logs/reports.
The wrapper waits until grading completes; use another terminal to tail the log.

`grading/summary.json` reports resolved, unresolved, empty patches, harness
errors, Pass@1, mean tool calls and finish count. A tool call here means one
`python_exec` invocation; it can contain multiple shell commands. Empty patches
remain in the selected-task denominator.
Missing/invalid official reports yield `pass_at_1: null` and a nonzero exit.
Generation errors also block grading. They are never presented as a clean score.
To retry grading after addressing an environment problem, use a fresh name such
as `--grade-dir grading-retry1`; do not reuse cached reports for changed patches.

The report can be recomputed without rerunning containers:

```bash
python scripts/eval_swebench_verified.py summarize --output outputs/swe-9b-contextgraph-smoke
```

## Validation status

The real HF preparation path was exercised successfully: 500 unique test tasks,
revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Full original agent loops
were exercised with scripted completions and a fake Docker backend, including
branch creation/return and patch export. A real temporary Git repository verifies
export of committed, staged, unstaged, new and deleted files.

Unit tests exercise public/gold separation, matched workflow selection,
container launch restrictions, environment lifecycle and tool dispatch, output
limits, patch-export errors, and official-report accounting. The Windows
development host has no Docker daemon; **real container creation, model-generated
patches and official test execution have not yet been validated here**. No
SWE-bench performance result is claimed. Start with the smoke run before a full
evaluation, and record Docker/runtime/model versions with the resulting artifacts.

References: [official evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/),
[pinned official harness](https://github.com/SWE-bench/SWE-bench/tree/3f01bd622c0a22c00406139f69a234ef08225f22),
[Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified).
