# SWE-bench Verified evaluation

This adapter generates repository patches with ReAct, FoldAgent, or isolated
ContextGraph, then evaluates those patches with the official SWE-bench harness.
It does **not** use the BC-P search corpus, answer matching, or an LLM judge.
No training is performed. Default model: `Qwen/Qwen3.5-9B`.

## Execution layout

### Single-task ARM pilot after a successful container preflight

An Apptainer adapter now supports **only `sympy__sympy-20590`** with the pinned
image used by the preflight. This is an ARM compatibility evaluation, not the
official x86 Docker environment or a full 500-instance Verified score.

The pilot now defaults to 65,536 total context tokens for both FoldAgent and
ContextGraph: 8,192 prompt tokens and 57,344 response tokens. Start a matching
model server on a GH compute node with
`SWE_MODEL_MAX_LEN=65536 bash scripts/serve_swe_qwen35_9b_vista.sbatch`.
The previously running 32K server cannot serve this pilot; stop it before
reusing its GPU and port, or use a free GPU and a different `MODEL_PORT`.
The server uses one GPU; additional allocated nodes do not combine GPU memory.
Check `/v1/models` for `max_model_len: 65536` before evaluation. Once it
responds, run the following in another shell on a compute node in the
allocation (set `SWE_ENDPOINT=http://SERVER-NODE:18000` if the server is on
another node):

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && SWE_AGENT_ENV=/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP bash scripts/run_swe_arm_pilot_idev.sh contextgraph
```

The runner installs the pinned upstream harness in a separate grading venv,
calibrates a clean baseline (must be unresolved) and reference patch (must be
resolved), generates a model patch using the existing code agent, and grades it
in a fresh workspace with upstream test scripts and parsers. Calibration's
reference patch and test scripts are never mounted in the generation sandbox.
Run the same command with `foldagent` to compare on the same task and budget.
`react` is also accepted. Set `SWE_CONTEXT_LENGTH=32768` to reproduce the
earlier 32K pilot, with a server configured for at least 32K.

The sandbox uses a network namespace with networking disabled and fails if the
site does not support it; there is no silent fallback to host execution/network.
Only a fresh task worktree is writable, mounted over `/testbed` so original image
Git refs are hidden. Per-task RAM/CPU cgroups are not implemented: allocation
limits apply, numerical libraries use one thread, and tool commands time out.

Artifacts: `$SCRATCH/context-graph-swe/runs/arm-pilot-JOBID-.../suite.log`,
`calibration/`, `generation/instances/`, and `generation/grading-arm/summary.json`.
`SWE_ARM_RUN_COMPLETE` means the one-task flow completed; inspect `resolved`
separately. Local tests cover adapters and real agent loops with mocked containers;
real ARM generation, calibration, and grading have also completed for this one
instance. Neither pilot result is an official x86 SWE-bench score.

### Prepare the Vista environment first

Run on the Vista login node:

For Python/data/tokenizer preparation directly with Bash, without an allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && bash scripts/setup_swe_verified_vista.sbatch
```

This mode uses one CPU thread for numerical libraries and installs wheels only
(no source compilation). It prints `SWE_VISTA_PYTHON_DATA_READY`, records
`image_preflight=pending`, and does not convert images or execute repository tests.
No Slurm job ID is required or fabricated. If a dependency wheel is unavailable,
installation fails explicitly. Without an allocation, this prepares only the
Python/data part of the environment, not the full ARM execution environment.

Data/tokenizer preparation uses a separate `preparation/` venv without inherited
Torch/CUDA, Ray, or TensorDict. Transformers 5 can import an installed Torch even
for tokenizer-only work; disabling `USE_TORCH` alone is insufficient. Package
versions are read from metadata, and `runtime_imports_checked=false` explicitly
records that agent runtime imports were not tested. Stage markers are flushed to
the setup log before metadata, tokenizer, and dataset checks.

After a failed setup, set `SWE_AGENT_ENV` to the existing `agent-...` directory
printed by pip or the setup log to reuse its installed dependencies. It must be
a venv under `$SCRATCH/context-graph-swe/envs`; the base RL environment is not
modified. The preparation venv is created inside it on the first retry.

The image/public-test probe can be run directly with Bash **on a compute node**
(for example inside `idev`). It does not need GPU computation. TACC login nodes
block Apptainer: direct Bash there is supported only for Python/data setup above.
Do not bypass the site wrapper or fabricate a Slurm job ID.

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && bash scripts/preflight_swe_apptainer_vista.sbatch
```

This downloads/converts one pinned SymPy ARM image, then runs one public test
file (600-second test timeout). Image compression uses one worker and a 256 MiB
squashfs memory setting; this is not a total process memory limit. Logs are
captured under `$SCRATCH/context-graph-swe/runs/preflight-direct-.../suite.log`.
The script rejects login hosts and checks actual image/test artifacts, because
the TACC login wrapper can print a refusal yet return exit code zero. This fixed probe
does not run an agent, start a model server, or grade generated patches.

To run both setup and the image/public-test probe inside an allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && mkdir -p logs && sbatch scripts/setup_swe_verified_vista.sbatch
```

This one-node `gg`, one-hour job prepares Python, pinned Verified data, the
existing Qwen3.5-9B tokenizer, and the SymPy ARM image/public-test probe. It
creates a fresh venv under `$SCRATCH/context-graph-swe/envs`, inheriting installed
packages from `deepseek_v4` to reuse ARM PyTorch without changing the RL environment.
Additional dependencies are installed into the new venv; its dependency snapshot,
Python executable, data path, and checks are recorded under
`$SCRATCH/context-graph-swe/runs/setup-JOBID-...`. This is an inherited environment,
not an independently reproducible lockfile environment.

Data uses revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` and is reused only
after validating public/grading checksums, task count, and probe base commit.
An incomplete existing data directory is rejected rather than overwritten.
The nested image probe reuses checksum-verified SIF images and records its own
`preflight-JOBID-...` directory. In batch mode, downloads/conversion/tests execute
inside the allocation. Logs: `logs/swe-env-setup.JOBID.out` and `.err`.

Success prints `SWE_VISTA_SETUP_COMPLETE`. This means environment preparation
and one public baseline probe passed; no model server or evaluation is launched.
The ARM single-task pilot above is a separate invocation. The Docker workflow
below remains the path for arbitrary Verified instances.

- **Model server:** one Vista GPU node, or an existing compatible vLLM server.
- **Agent runner and repository containers:** a machine with access to a Linux
  **x86_64 Docker daemon**. CPU-only is sufficient here; inference is remote.
- **Official grader:** a separate Python environment on the same Docker host.
  Keep its instance images until grading is finished.

Vista ARM nodes are not supported as repository-container hosts by this initial
Docker adapter. Use the explicit single-task Apptainer pilot above on Vista.
Do not submit the default Docker generation/grading script there expecting `sbatch`
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

### Vista ARM image preflight in a batch job

On the Vista login node, submit this independent **one-node CPU (`gg`), one-hour**
probe; it does not use or restart the model server:

```bash
mkdir -p logs
sbatch scripts/preflight_swe_apptainer_vista.sbatch
```

The `AST24021` account must have access to the selected partition; a `gh` allocation
can instead be selected with `sbatch --partition=gh` if needed. A rejected submission
does not run the probe. Slurm logs are `logs/swe-arm-preflight.JOBID.out` / `.err`.
The log directory must exist **before** submitting.

The probe pins Epoch's `sympy__sympy-20590` ARM64 image by OCI digest
`sha256:a8b2a5265717391b168a5d7aa884b466e80fa748d074373a56d328cc48d887b9`.
Registry metadata was checked: Linux ARM64, approximately 0.90 GB compressed.
The task and base commit were checked against Verified revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.

Images and their SIF checksums live under `$SCRATCH/context-graph-swe/images`;
the Apptainer cache and conversion temporary files also stay on `$SCRATCH`.
Each run has a fresh `$SCRATCH/context-graph-swe/runs/preflight-JOBID-...` directory
containing the full log, provenance and a disposable writable repository copy.
Concurrent image pulls are locked, and cached SIF files are checksum-checked.
No old artifacts or running services are removed.

The fixed probe checks ARM execution, the exact base commit, activation of the
image's `testbed` Python environment, import of the writable SymPy checkout, and
existing public tests in `sympy/core/tests/test_basic.py` (10-minute test timeout).
It loads no gold/test patch and executes no agent-generated commands. A successful
run prints `SWE_APPTAINER_PREFLIGHT_COMPLETE` and writes `preflight-complete.txt`.
This is **not** an issue-resolution test or an evaluation score. Neither this
probe nor Alpine success proves compatibility of all 500 tasks. Epoch labels its
ARM images best-effort and untested. The public baseline passed 22 tests on
Vista job 1035198; model-generated patch execution and grading remain to be validated.
See the [image provider](https://github.com/epoch-research/SWE-bench).

The Docker generation/grading workflow below remains unchanged. A production
Apptainer backend still needs writable per-task repositories, isolation for
agent commands, patch export, and compatible test execution/report parsing.

### Start the inference service

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
# ARM grading environment repair

The initial ARM pilot merged container stderr into exported patches and attempted
an editable install into the read-only SIF environment. Patch extraction now keeps
stdout separate. Grading copies `/opt/miniconda3/envs/testbed` into a fresh private
scratch directory and binds that copy back at the same prefix. The copy is removed
when grading finishes; the shared image and agent environment are unchanged.
This uses Apptainer [writable bind mounts](https://apptainer.org/docs/user/1.1/bind_paths_and_mounts.html).

The pinned upstream evaluation script is saved as `upstream_eval.sh`. Its executed
copy enables shell error checking during setup and verifies that SymPy imports from
`/testbed` using the testbed Python prefix. Error checking is disabled at the upstream
test-output boundary so failing tests still produce complete grading logs. Setup or
import verification failures abort grading rather than count as unresolved tasks.
Grading also creates a private HOME under the container's isolated `/tmp`, so
`git config --global` and package configuration do not reference the unavailable
host home. Setup failures include the log tail in the error output.

Before another generation run, use the existing compute allocation to calibrate
only (no model endpoint or GPU inference needed):

```bash
cd /work/09281/chc_1996/vista/context-graph && SWE_AGENT_ENV=/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP bash scripts/run_swe_arm_pilot_idev.sh calibrate
```

Require `SWE_ARM_CALIBRATION_COMPLETE` and inspect the new calibration logs for
successful installation and `SWE_ARM_IMPORT_OK /testbed/...`. Earlier calibration
results do not validate this repaired installation path. Local tests cover shell
failure propagation, mount isolation, and patch extraction; Vista execution is
still required to validate the writable environment copy on the actual image.
