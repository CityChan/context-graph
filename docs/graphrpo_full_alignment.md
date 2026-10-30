# Full graph-management RL recipe

`graph_management_full_v1` is the default in
`scripts/train_bcp_qwen35_9b_guarded_rl.sh`. It implements the functionality and
numeric settings shown in the supplied branch-comparison screenshots. The source
of `huanchen/master-rocm` at `fb971a9` was unavailable during implementation;
this is not a claim of source-identical reproduction. Unspecified rules are listed
below so they can be compared when that source becomes available.

Execution and graph management are jointly optimized with GraphRPO. The model
uses explicit task and graph tools, with thinking before a typed JSON action.
The native RL transport does not constrain reasoning with a whole-response JSON
grammar. Final JSON is schema-validated before dispatch; native reasoning, EOS,
sampled input IDs and rollout logprobs remain authoritative training data.
Malformed final actions execute no tool. There are no extra QA probes, M-only
training phases, or alternating E/M updates in this recipe.

## Alignment map

| Difference | Full recipe |
| --- | --- |
| Terminal shaping | Forced-finish discount, graph score and request-token efficiency |
| Local shaping | Effective-operation credits and the process penalties below |
| Negative advantages | `negative_all`, including labeled environment actions |
| Model edge density | Cumulative successful model edge additions, including subsequently inactive edges |
| Training context | Existing native-token/logprob/mask audits and lossless request splitting retained |
| Automatic maintenance | Optional auto-prune and auto-compact, enabled by this launcher |
| Compact snapshot | 1,500 characters per node by default; full stored content remains readable |

Older profiles `graph_management_v1` and `invalid_action_v1` remain selectable.
Their defaults do not acquire the full recipe's new shaping or automatic
maintenance. Existing unrelated launchers keep their own defaults.

## Terminal reward and local credit

`R = r' + 0.2 G + 0.3 r' max(0, 1 - T / 3000000)`

- `r'`: naturally correct = 1; correct after harness finalization = 0.5;
  incorrect = 0. A failed finalization does not manufacture a correct result.
- `T`: all model-request input plus output tokens across main and branches.
- `G`: zero below four observation nodes; otherwise
  `0.4 archive_ratio + 0.4 size_score + 0.2 density_score`.
- `archive_ratio`: inactive observation nodes / all observation nodes.
- `size_score = min(1, 12 / max(active_nodes, 1))`.
- `density_score = min(1, cumulative_model_edge_additions / max(active_nodes - 1, 1))`.
  Only new edges from successful explicit `add_edge` calls count. Framework
  edges do not. The numerator persists when endpoints later become inactive.

Native task reward remains separate for accuracy reporting. Validation/evaluation
reports native task reward, not this shaped training score. Wrong answers can
receive up to 0.2 graph shaping. No separate overflow penalty is added.

The previous [effective-operation credits](graphrpo_graph_management.md) remain:
merge/prune that reduce active nodes +0.1; a new model edge +0.05; model compact
that saves at least 2,000 tokens with pre-call occupancy at most 75% +0.2.
Compact earns at most two bonuses; all positive local credits share a 1.0 cap
across main and branches. These credits are `C`, not additions to terminal `R`.
Reads, selects, recalls, ineffective calls and framework maintenance earn no
positive local credit.

## Process labels and advantages

| Event | Raw process label `P` and target |
| --- | --- |
| Executor-rejected graph action | -0.3 on that action |
| Repeated identical normalized query/page | -0.2 on the repeat |
| Highly similar branch assignment | -0.3 on the delegation |
| Fifth and subsequent branch attempts | -0.1 on the delegation |
| Forced branch return | -0.1 on parent delegation and branch return; exempt if delegated at >=75% occupancy |
| >12 active nodes and >10 environment actions since effective model edit | -0.1 on the environment action |
| Environment action with input occupancy >=90% | -0.05 on that action |
| High pressure within four environment actions after an edit | -0.2 on the original edit |
| A merged/pruned observation's query/page retrieved again | -0.2 on the original edit |
| Main retrieval result >=1,000 estimated tokens | `-min(0.2, result_tokens / window)` on that retrieval |
| Successful episode, >8 environment actions after observing an answer string | -0.1 on each later environment action |

For overlapping penalties on an action, use the most negative label, not their
sum. Metadata records each reason and its value. Scope-process shaping is off.
Answer matching only affects training labels; answers are never injected into
policy prompts. Matching inspects visible appended feedback after any budget
truncation, not hidden raw result text.

With `alpha=beta=1` and group mean-centering without division by standard deviation:

`A0 = R - mean_group(R) + C + P`

On every policy token with `P < 0`, `A = min(A0, P)`; otherwise `A = A0`.
Padding, historical conditioning tokens and excluded rollout rows receive no
such signal. Tied-outcome groups remain eligible for dynamic sampling if they
contain a nonzero credit or negative process label. Native context and token
position audits also apply after automatic compact.

## Automatic maintenance and snapshots

The full launcher enables the following optional framework behavior. Automatic
events are tagged `source=framework`, create no policy action, and earn no local
operation credit. They can still change the terminal graph score and rollout
outcome, so evaluation must record whether they are enabled.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `GRAPH_MANAGEMENT_AUTO_PRUNE_MAX_ACTIVE` | 12 | Existing low-value auto-prune heuristic; 0 disables |
| `GRAPH_MANAGEMENT_AUTO_COMPACT` | True | Enable automatic context replacement |
| `GRAPH_MANAGEMENT_AUTO_COMPACT_THRESHOLD` | 0.75 | Current context plus pending observation estimate / window |
| `GRAPH_MANAGEMENT_AUTO_COMPACT_COOLDOWN` | 4 | Minimum environment actions between automatic compacts |
| `GRAPH_MANAGEMENT_AUTO_COMPACT_MAX` | 6 | Shared episode limit across main and branches |
| `GRAPH_MANAGEMENT_AUTO_COMPACT_MIN_GAIN` | 0.25 | Minimum predicted fractional reduction of existing context |
| `GRAPH_COMPACT_NODE_CHARS` | 1500 | Per-node snapshot content cap; 0 disables |
| `GRAPH_MANAGEMENT_BRANCH_SIMILARITY` | 0.8 | Branch duplicate threshold |

Maintenance runs before ordinary generation and after tool results, with pending
observation size included in the pressure trigger. Forced finalization does not
start new automatic maintenance. Compact preserves task/branch instructions and
active graph state, starts an immutable context segment, and retains the native
policy history for training. Truncation is marked in snapshot nodes and does not
modify the stored graph: `graph_read` can page through the full raw content.
Repeated identical retained context is rejected for both model and automatic
compact, per agent. A failed or low-gain automatic compact earns no credit.

## Explicit choices where screenshots omit implementation details

The size and density normalizations above, and the large-main-result ratio, are
documented implementation choices rather than verified old-source formulas.
Duplicate branch similarity uses Jaccard overlap of case-folded whitespace words
in description + prompt (default 0.8). Retrieval equality uses case-folded,
whitespace-normalized query, docid or URL; it does not resolve URL/docid aliases.
Repeated-query, branch and environment-action counters are episode-wide, shared
across agents. Cooldown and delayed labels use that environment-action clock.
An effective edit resets the neglect clock only for successful state-changing
merge/prune/add_edge or successful compact. Framework maintenance does not reset
that model-edit clock. Answer detection uses normalized, word-bounded string
matches; it is a proxy, not proof that evidence establishes the answer. Large
result size is measured before observation budgeting. Local positive credits
and these structural proxies do not certify semantic edit quality.

## Launch and verification

In an existing four-node Vista allocation (one search node, three trainers):

```bash
bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

This defaults to a 64K working context and one training step. For the existing
five-node, 50-step recipe:

```bash
SMOKE_TEST=0 bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

To run full shaping with model-owned maintenance only:

```bash
GRAPH_MANAGEMENT_AUTO_PRUNE_MAX_ACTIVE=0 GRAPH_MANAGEMENT_AUTO_COMPACT=False bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

`training-config.txt`, resolved Hydra config, graph traces and `env_stats` record
the selected recipe, automatic settings, penalty reasons, terminal components
and credit caps. Confirm `graph_management_full_v1`, `negative_all`, 64K and
thinking enabled before interpreting results. Compare raw task accuracy,
request-token consumption and operation statistics separately from shaped reward.
Local tests cover reward math, penalties, dynamic groups, native thinking tokens,
automatic maintenance, context exports and real launcher/Hydra composition.
They do not establish GPU logprob parity, training convergence or task improvement.
