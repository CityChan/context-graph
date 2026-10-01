# Codebase map and cleanup boundaries

## Runtime methods

| Method | Search/interactive executor | Code executor |
| --- | --- | --- |
| ReAct | `agents/react_agent.py` | `agents/react_agent_code.py` |
| FoldAgent | `agents/fold_agent.py` | `agents/fold_agent_code.py` |
| ContextGraph, isolated | `agents/graph_agent_isolated.py` | `agents/graph_agent_code_isolated.py` |
| ContextGraph, global | `agents/graph_agent.py` | No separate global code executor |

`scripts/eval_gaia.py` maps `search`, `search_branch`, and `search_graph` to
ReAct, FoldAgent, and isolated ContextGraph. `scripts/eval_interactive.py`
uses the same three executor families. `scripts/train_graph.py` still registers
both global and isolated graph loops. The global loop is therefore not dead code.

The isolated graph loop's `legacy`, `repaired`, and `foldagent` modes are
experimental controls, not duplicate implementations to delete. StructMem is
currently an optional `structured_memory.py` component, not an independent
official StructMem executor. GraphRPO is training credit assignment.

## Shared implementation

| File | Responsibility |
| --- | --- |
| `agents/agent_text.py` | Last XML tool call, summary extraction, branch fallback text |
| `agents/utils.py` | Agent context, model adapters, common trajectory formatting |
| `agents/context_graph.py` | Graph state, archive retrieval, graph operations |
| `agents/graph_operations.py` | Isolated search/code graph tool validation and responses |
| `agents/graph_controller.py` | Structured graph decisions |
| `agents/finalizer.py` | Protected final-answer budget and forced completion |
| `agents/structured_memory.py` | Fact memory and knowledge-gap analysis |
| `scripts/audit_records.py` | Shared JSON/JSONL discovery and loading for audit tools |
| `scripts/evaluation_records.py` | Three-shard provenance and outcome validation |
| `scripts/export_audited_results.py` | Sanitized publication of reconciled saved results |
| `envs/judge_client.py` | Judge request retries and HTTP-client lifetime |
| `envs/async_results.py` | Search-result delivery on the owning event loop |

Existing helper import paths on executor and audit modules remain available
through imports. Global graph handlers remain separate where validation order or
response text differs: those strings are model observations, so changing them
would change the evaluation protocol.

## Other directories

- `envs/`: benchmark environments, tool clients, search servers and code sandbox.
- `scripts/`: data preparation, training/evaluation entry points, Vista launchers,
  and offline audits. Benchmark/model/allocation variants carry real configuration
  differences; similar filenames alone do not justify deleting a launcher.
- `tests/`: local unit, protocol, and integration checks. Some require Ray, Torch,
  cached tokenizers or external benchmark environments.
- `verl/`: vendored training framework with project changes; do not replace it
  with an upstream package as part of duplicate-code cleanup.
- `logs/`, `outputs/`, `results/`, `memory_data/`: run evidence and local artifacts;
  they are not cleanup targets for source refactoring.

The obsolete `external/verl` submodule declaration was removed: there is no
tracked gitlink at that path; the framework lives in `verl/`. The empty root
file `0.11.0` was also removed.

## Validation and deployment

The October 1 review removes the unused proxy-judge duplicate and the five
broken session-restart blocks. `enable_summary=True` now raises before opening
an environment; it is not a supported memory-compression feature. Branch-return
and graph-merge summaries are separate and retained.

Global/isolated variants, the code/search environment parsers, launch wrappers,
and `verl` remain separate where behavior or configuration differs. Similar
filenames alone are not evidence of redundant code.

See the [review report](project_review_20261001.md) for final test results,
known limitations and behavior changes. In particular, answer acceptance and
scope-judge failure handling changed; this is not a protocol-neutral refactor.
Do not update a shared Vista checkout while its jobs are running.
