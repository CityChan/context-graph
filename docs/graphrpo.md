# GraphRPO implementation

This implementation follows Section 4.2 of the ContextGraph manuscript. It is
enabled only when `algorithm.adv_estimator=graphrpo` and
`actor_rollout_ref.actor.policy_loss.loss_mode=graphrpo`.

## Training semantics

- A `gen_uid` identifies one episode. Its main and branch streams share the
  verified binary terminal reward and one episode-level outcome advantage.
- The group baseline uses the population standard deviation. A zero-variance
  group receives zero outcome advantage, and every question must contain at
  least two non-dummy episodes.
- Valid policy-generated `merge`, `prune`, `add_edge`, and `select` decisions
  receive the clipped frozen-evaluator utility increment on successful
  episodes. `pass`, automatic edits, failed edits, and failed episodes receive
  no graph increment.
- Edit increments and non-positive process labels are broadcast across their
  generated assistant-turn spans. Multiple process labels use the most
  negative applicable value.
- The actor loss is weighted first uniformly over all policy tokens in an
  episode, then uniformly over episodes for a question, then over questions in
  the global batch. The same weights are used for entropy and low-variance KL.

GraphRPO deliberately requires the structured graph controller. This makes one
controller response exactly one edit decision span and prevents legacy XML
graph calls from being mixed with environment actions.

## Frozen graph evaluator

`scripts/prepare_graph_evaluator_data.py` extracts question-disjoint binary
training data from saved rollout result JSON. `scripts/train_graph_evaluator.py`
trains a sequence classifier and fits a scalar validation temperature.
`scripts/serve_graph_evaluator.py` loads that checkpoint in evaluation mode and
serves the `contextgraph.graph_evaluator.v1` protocol at `POST /score`.

The policy launcher never substitutes a heuristic score when the evaluator is
missing or unavailable. Set `GRAPH_RPO_EVALUATOR_URL` to the frozen service's
`/score` endpoint before launching
`scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh`.

The evaluator checkpoint and calibration must be fixed before a policy run.
Judge outputs used for task outcomes or scope labels should likewise be cached
or generated with a fixed decoding configuration for reproducible experiments.

For mechanics-only validation on an existing five-node allocation,
`scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh` starts a deterministic
CPU evaluator on the search node and performs one optimizer step. Its evaluator
scores are deliberately synthetic: use the smoke only to verify wiring and
never include its reward or checkpoint in experiments.
