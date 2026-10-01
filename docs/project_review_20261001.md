# Project review — 2026-10-01

Baseline: `6c17c83ba211704f20f752ba5f16b5fa89e93ec1` on `master`.
Scope: agent execution/finalization, search and judging, graph controller and
reward semantics, evaluation/grading evidence, launch profiles, module layout,
and the local regression suite. This is a source/artifact review, not a claim
that every GPU, container or distributed execution path is defect-free.

## Confirmed fixes

| Finding | Previous behavior | Change / validation |
| --- | --- | --- |
| Finalizer accepted research as an answer | Removed XML tags and scored the remaining branch/search arguments or thinking; a finish anywhere in a mixed response counted as valid | Strip explicit reasoning, reject research/mixed tool output as an answer, accept a single finish or visible plain prose; regression tests cover truncated reasoning and mixed calls |
| Finalizer retried rejected finish | Repeated an identical finish to clear environment guards | Submit once and respect rejection |
| Premature search-environment completion | Empty answers, must-search rejection and double-check set `is_finish` before acceptance | Set completion and store the answer only after all guards accept |
| Reward without answer submission | Guessed an answer from the last assistant turn even after no finish/rejection | Score only an accepted finish; otherwise return zero without calling the judge |
| Search requirement depended on gold answer | Only an exact reference match triggered the must-search guard; malformed search calls counted as search | Enforce the guard independently of the reference answer and only after a search request succeeds |
| Scope judge service failures caused penalties | Missing/unparseable API judgment was treated as out-of-scope | Only explicit `<error>` earns the scope penalty; service/format failure returns neutral with an ungraded marker |
| Unclosed judge clients | Per-call SDK clients were never explicitly closed | Shared context-managed judge client, tested on success, failure and cancellation |
| Search-client lifetime at reward collection | Per-environment HTTP pools remained open after scoring | Close in reward collection's `finally`, including scoring exceptions |
| Cross-thread asyncio result delivery | Search collector thread called `Future.set_result` directly | Schedule on owning loop with `call_soon_threadsafe`, check cancellation inside callback; debug-loop/cancellation tests |
| Pending cancelled searches | Cancellation could leave entries in `pending_requests` | Remove entries in `finally`, in addition to completion/timeout cleanup |
| Broken session restart | Five executors made `main+` summaries but continued stepping `main` | Remove the unusable repeated blocks; reject `enable_summary=True` before environment initialization |
| Inconsistent result aggregation | Aggregation assumed nonempty selection and did not check per-rank placement or binary task rewards | Shared validator checks selection, rank, provenance, row placement and task outcomes; exporter also reconciles stored summaries |

The empty-select graph-controller crash was already fixed in baseline commit
`6c17c83`: arity is validated before indexing or converting active selection
to pass. Its regression tests remain in the suite. The posted SWE run reported
zero controller errors, so that bug is not established as its failure cause.

These fixes change answer/reward behavior. Do not compare new scores to old
ones as a pure memory-method intervention. The published historical results
retain their original code commits and were not rescored.

## Cleanup and organization

- Removed the unused duplicate proxy judge and consolidated direct judge requests.
- Removed five ineffective session-restart blocks. This disables an unsupported
  option explicitly; it does **not** implement safe session rebasing.
- Shared live/offline evaluation validation in `scripts/evaluation_records.py`.
- Relocated the unchanged Ray-environment log redactor into
  `verl/utils/logging_utils.py`, retaining its training-entrypoint alias. Its
  pure CPU test no longer imports GPU worker engines.
- Replaced the long root README with current workflow/status/results navigation;
  preserved detailed model-specific recipes in `docs/workflows.md`.
- Added a results index and a reproducible sanitized artifact exporter.
- Updated two stale tests that expected an unlimited-thinking instruction and
  the old graph-state injection default. Current budget and memory-mode
  behavior are asserted instead; production prompts were not changed here.

Global/isolated graph executors, code/search parsers, legacy/repaired memory
controls and model/topology wrappers have real behavioral differences and are
retained. The vendored `verl` tree is not redundant with an external package.
The existing untracked Qwen3-235B teacher launcher is outside this change.

## Remaining decisions and limitations

1. **Session summary restart is unsupported.** Re-enabling it needs explicit
   active-session switching, graph turn-index rebasing, aggregate budget and
   training-trajectory accounting tests. Branch-return summaries and graph
   merge summaries are unaffected by this guard.
2. **Graph cost is success-gated.** `compute_graph_reward` applies
   `lambda_cost * cost` only when task reward is positive. The current
   ContextGraph launcher uses `lambda_cost=0.02`; this can make equal-cost
   success/failure trajectories receive asymmetric shaping. It is a protocol
   decision, not silently changed to a different RL objective in this cleanup.
   “Always Asking” descriptions should not be read as a verified match to this
   precise reward formula.
3. **Branch budgets include inherited history.** A branch copies main history
   and uses the existing response/context accounting. It does not automatically
   receive a fresh full response budget on top of that history. Changing this
   changes method cost and comparison fairness; no fresh-budget claim is made.
4. **Training and evaluation memory differ.** Training retains generated
   history for token/log-probability alignment. Graph pruning/active-node count
   alone does not prove a reduction in the policy's training working context.
5. **Five-node 32K RL is not GPU-validated here.** The 4 x 4 launch profile and
   arguments are tested, but optimizer updates, weight sync, memory peak,
   checkpoint reload and 48-hour stability need a real allocation. An active
   search log or zero-percent update bar cannot prove either deadlock or progress
   through an optimizer step.
6. **Resource cleanup before reward collection remains incomplete.** The
   search-client close now covers normal reward collection and judge failures.
   A model/executor exception before reward collection still needs an outer
   environment-lifetime cleanup path; this review does not claim that all
   cancellation paths across all benchmark environments are covered.
7. **Official SWE evaluation is outstanding.** The saved Verified record is
   one empty-patch ARM pilot. Later 64K Lite runs are console evidence only in
   this checkout. No full official x86 Verified or Lite score is established.

## Evidence publication

The [audit bundle](../results/audited/README.md) reconciles eight complete
BC-P/local-GAIA runs and records six exclusions. It publishes answer-free
per-instance outcomes, settings, exact commits, dataset/checkpoint identity and
source-file hashes. Older result summaries remain a separate historical tier.

The synced `swe-foldagent-1035199-hgEsOA` files agree on prediction SHA-256,
empty patch, selected task, image identity metadata and completed ARM grading:
0/1 resolved. The image/dataset payloads and calibration report are not present
locally, so their recorded hashes/calibration are not independently revalidated.

## Validation

Final local suite: **653 passed, 5 skipped**, no failures (26.43 seconds).
Executed with `python -m pytest -q tests --tb=short -rs` in an isolated
Windows Python 3.11 environment with the existing CPU dependencies plus Ray
2.58.0, cloudpickle 3.1.2, TensorDict 0.10.0, codetiming 1.4.0,
hydra-core 1.3.7 and torchdata 0.11.0. This is a CPU test environment, not
the pinned Vista training stack. The suite has one Ray API deprecation warning.

Skipped: one cached-Qwen tokenizer check (`QWEN_TOKENIZER_PATH` unset), and four
PyArrow/Parquet checks because PyArrow is not installed in this environment.
Earlier baseline collection failed on missing Ray/cloudpickle; after partial
collection it had 568 passing tests, 20 Ray-import failures, two stale assertion
failures and eight skips. The missing dependencies and stale assertions were
addressed without replacing GPU execution with a claimed live training result.

Additional checks:

- Python AST syntax: **527 files passed**, including the vendored Python tree.
- Bash syntax: **219 tracked `.sh` / `.sbatch` files passed** with `bash -n`.
- Updated README/code-map/result navigation: relative link targets exist.
- `git diff --check`: passed.
- Published result totals reconcile with their per-instance rows; selected
  output contains no credential fields, raw answer/message fields, or absolute
  local cluster paths.
- Re-export from the saved inputs reproduces both publication files byte for byte.

CPU regressions and mocked launcher tests are distinct from live model, GPU,
distributed-update and container integration. No new TACC job was submitted or
cancelled during this review.
