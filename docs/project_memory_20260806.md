# ContextGraph Project Memory — 2026-08-06

This note preserves the current project conclusions and implementation state
for future sessions. Treat it as a snapshot, not as a substitute for checking
the current code and experiment logs.

## Active benchmark tracks

- BrowseComp-Plus: primary search/research training and evaluation benchmark.
- GAIA text-only: active transfer/evaluation track; file-attachment tasks are
  excluded by the current search-only integration.
- ScienceAgentBench: active code-agent evaluation track, with separate code
  agent loops.
- ALFWorld: experimental/diagnostic until the corrected official harness is
  validated end to end on TACC.

HotpotQA, MuSiQue, and 2WikiMultiHopQA are no longer active integrations.

## GAM comparison and positioning

GAM is a training-free, cross-session long-term dialogue memory system. It
separates a local Event Progression Graph from a global Topic Associative
Network, consolidates at semantic boundaries, and retrieves top-down from
topics to archived raw events.

ContextGraph should be positioned differently:

> A learned working-context controller for long-horizon tool agents, using
> spatially isolated subtask graphs and task-optimized evidence retrieval.

The defensible distinction is agent control and learned context management,
not merely hierarchical graph memory. Hierarchical local/global graphs,
summary plus raw evidence, state switching, and graph-guided retrieval overlap
substantially with GAM and should be cited as related work rather than claimed
alone as ContextGraph novelty.

## Retrieval-memory implementation

The current worktree adds a GAM-inspired prototype:

- Detached child graphs can be retained as raw evidence archives.
- Parent summary nodes store archive and evidence pointers.
- `ContextGraph.retrieve_context()` produces a bounded lexical retrieval view
  with separate summary and raw-evidence budgets.
- Pruned evidence remains recoverable.
- The search ContextGraph agent archives full tool observations, retrieves
  relevant memory, and logs archive/retrieval metrics.
- The main 8B BrowseComp ContextGraph script exposes retrieval-memory budgets.

Default prototype budgets are 768 summary tokens, 1280 evidence tokens, at
most five summaries and four evidence items.

The implementation currently exists only in the search isolated agent path;
the code/SAB isolated agent has not been unified with it.

## Critical training-semantics risk

Do not train the current in-place working-context replacement path without
fixing trajectory semantics.

`graph_agent_isolated.py` currently calls `Agent.replace_user_turn()` after
later assistant completions may already have been generated. Consequently, a
completion can be generated from the original full observation but later be
recomputed during RL against an archive marker. The rollout prompt and the
training prompt/log-probability context are then different.

Safe short-term policy:

- Use retrieval-memory replacement for zero-shot/evaluation only.
- Disable in-place replacement during training, or retain full training chat
  while logging hypothetical compression metrics.

Preferred long-term fix:

- Use immutable memory sessions. At a consolidation boundary, end the current
  Agent trajectory and begin a new one from the original task plus a bounded
  retrieved-memory snapshot.
- Add a regression check that the prompt hash used for generation equals the
  prompt represented in the training sample.

## Recommended architecture refactor

Split the current monolithic graph implementation into:

```text
agents/memory/
  schema.py          # typed nodes, evidence references, relations
  store.py           # active graph and detached archive
  consolidation.py   # branch/event boundary policies
  retrieval.py       # recent, lexical, embedding, graph retrieval
  rendering.py       # token-budgeted working-context views
  metrics.py         # retrieval/compression measurements
```

Unify duplicated search/code agent loops around a shared runner:

```text
AgentRunner
  ContextPolicy: Linear | Fold | GraphMemory
  DomainAdapter: Search | Code | ALFWorld
```

This is necessary because search and code ContextGraph variants currently
drift when features are added to only one copied loop.

## Retrieval and consolidation roadmap

The lexical retriever is a dependency-free baseline, not the intended final
method. The next retriever should use:

1. a structured `MemoryQuery` containing task, current subtask, entities,
   unresolved claims, and temporal constraints;
2. embedding retrieval over consolidated summaries;
3. one-hop graph expansion;
4. archive drill-down to raw evidence;
5. reranking by semantic relevance, provenance, confidence,
   support/contradiction, and token cost.

Consolidation should be event-triggered at branch return, claim resolution,
subtask completion, semantic shift, or context-budget pressure. Fixed
`consolidation_interval` should remain only as a fallback.

## Reward guidance

Do not treat edge count, graph density, or explicit graph-operation count as
evidence that memory helped the task. Existing graph shaping can reward
failed trajectories and is vulnerable to proxy optimization.

Recommended experiment order:

1. evaluate retrieval architecture with task reward only;
2. log graph/retrieval metrics without optimizing them;
3. add retrieval-oriented rewards only after zero-shot benefit is established.

Candidate useful signals are evidence recall, summary fidelity, raw-evidence
recovery, verifier improvement over recent-context retrieval, irrelevant
retrieval rate, and actual generation-context tokens.

## ALFWorld diagnosis and correction

The previous conclusion that Qwen3 8B simply cold-started because ALFWorld was
too hard was premature. The custom harness bypassed ALFWorld's official
`AlfredDemangler`, manually truncated coordinate-bearing entity IDs, and sent
the truncated command directly to TextWorld without a reverse mapping. It
also scanned games without the official solvability/unsupported-task filters.

The current worktree corrects this by:

- registering TextWorld games with official `AlfredDemangler(shuffle=False)`
  and `AlfredInfos` wrappers;
- displaying and executing the exact official-demangled admissible commands;
- rejecting non-exact commands in `@real` mode before environment execution;
- exposing only the `action` environment tool in ALFWorld prompts;
- counting only real `TextWorld.step()` calls toward the environment budget;
- filtering generated parquets to `solvable=true`, requiring
  `traj_data.json`, and excluding movable/Sliced tasks;
- disabling admissible-command truncation by default.

The corrected harness has unit coverage but still requires a real TACC
end-to-end check. Regenerate ALFWorld parquets before that check because old
parquets were produced with the previous scanner.

## Verification state

At the time of this note:

- nine local tests pass;
- relevant Python modules compile;
- modified BrowseComp and ALFWorld shell scripts pass `bash -n`;
- `git diff --check` passes;
- the retrieval-memory and official-ALFWorld changes are still uncommitted in
  the working tree.

## Immediate priority order

1. Protect RL training from in-place history mutation.
2. Validate the official ALFWorld harness with an oracle/expert and an 8B
   action-only `@real` smoke.
3. Run BrowseComp retrieval-memory on/off zero-shot evaluation.
4. Refactor memory storage/retrieval and unify the agent loops.
5. Add semantic retrieval and event-triggered consolidation.
6. Only then train retrieval/consolidation policies and redesign rewards.
