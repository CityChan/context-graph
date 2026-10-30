# Explicit graph context distillation

This is a separate SFT profile, `explicit_context_distillation_v1`. Existing
Route A auto/controller collection and assisted-teacher exports remain available.
This implementation does not import the unshared `huancheng/graphrpo-compress-r1`
branch or its ROCm launcher.

The teacher can be an exported RL checkpoint, such as step60. Collection uses
explicit tools with thinking: the model chooses both operation timing and content.
There is no periodic controller, automatic pruning, or automatic history replacement.
Temporary system guidance encourages useful operations without imposing quotas.
The native-completion transport saves actual input/output IDs, reasoning, EOS,
messages, schemas, model identity and graph traces. Reasoning is generated freely;
the final JSON is validated against the action schema before execution. Invalid or
unfinished episodes are not positive SFT examples. The server must support native
completion `token_ids`; a preflight checks the explicit thinking/action path.

The exporter removes only the exact registered system guidance block. It preserves
task text, observations, graph state and model answers in the saved messages, then
renders the student prefix with thinking enabled. Original teacher output IDs,
including reasoning and EOS, remain the targets. Re-rendered prefixes can differ
from native teacher prefixes beyond the removed text because conversation rendering
and incremental token history differ. The export therefore records both hashes and
sets `exact_replay=false`: this is context distillation, not on-policy replay.

The student may be the original base checkpoint instead of the teacher's RL
checkpoint. Vocabulary, special-token and template fingerprints must match. The
current path does not support cross-tokenizer distillation. Source checkpoint
provenance hashes configuration/index files, not the large weight shards; keep the
source checkpoint immutable. Legacy message-only artifacts need a separate audited
adapter and are deliberately rejected rather than relabelled as native-token data.

## Data gates and training

- Only successful, naturally finished episodes with clean protocol/graph statistics
  and consistent sampled graph traces are eligible. Node references must occur in
  the retained visible context and the recorded graph state. This is a reference
  visibility check, not proof that every retained summary preserves all evidence.
- Source training tasks are checked against the supplied independent evaluation
  parquet **before** sampling or success filtering. Splitting is deterministic by
  normalized question (ScienceWorld: task and variation), across candidates/runs.
  Exact normalized identity checks do not detect paraphrases or all semantic leakage.
- Multiple collection directories can be merged. Overlapping supervised token
  segments are discarded as units; source histories are never spliced across tasks.
- Graph-tool episodes and no-graph-tool controls remain separate exports. The mixed
  train file caps controls at 0.5 rows per graph row by default. Read-only graph calls
  count as graph-tool episodes; operation coverage reports distinguish them from edits.
- `loaded_coverage` reports actual selected rows, distinct questions, source episodes,
  prune episodes and operation calls after deduplication/mixing. Calls and segments
  are not independent trajectories. No prune oversampling is introduced.
- No semantic edit reviewer is used in this profile. A successful final answer does
  not prove every prune/merge beneficial; `evidence_review_performed=false` records this.
- `agent_context` loss masks supervise only recorded targets. History after compaction
  is conditioning only, unless its sampled prefix matches exactly within a segment.
  Overlength requests fail instead of truncating inputs. All exported train and
  validation rows, hashes, tokenizer signatures and heldout isolation are checked
  again before training.

Collection and training default to **65,536 input + output tokens**. Training is
full-parameter assistant-target SFT by default (`LORA_RANK=0`), not RL. The training
launcher defaults to one optimizer step, batch4/microbatch1, LR1e-5: a mechanics
smoke, not a tuned full training recipe. Set steps/epochs deliberately for a longer
run. No GPU memory-fit or benchmark improvement claim follows from local tests.

## Vista usage

Use an idle four-node idev allocation. This reuses the existing one-GPU Qwen server,
optional BC-P retrieval server and CPU collector; training uses four GH200 GPUs.
It does not reproduce the screenshot's 32-MI300X setup. Keep checkpoints and outputs
under `$SCRATCH`. Preserve the existing `cxtgraph` collector and `deepseek_v4` training
environments. Set these paths to real local artifacts before invoking the commands:

```bash
export DISTILL_TEACHER_CHECKPOINT=/absolute/scratch/path/to/rl-step60-hf
export SFT_STUDENT_MODEL_PATH=/absolute/scratch/path/to/original-qwen35-9b
export TRAIN_DATA=/absolute/path/to/bcp-training.parquet
export HELDOUT_DATA=/absolute/path/to/independent-bcp-evaluation.parquet
SAMPLES=3 CANDIDATES=2 bash scripts/generate_explicit_distillation_sft_4node_idev.sh bcp
```

The shared server advertises `Qwen/Qwen3.5-9B` by default, while loading the explicit
teacher path; the collection manifest records the checkpoint separately. Supported
server IDs follow the shared Qwen launcher. ScienceWorld is available by replacing
`bcp` with `scienceworld`; that launcher generates official train/heldout variation
files under its run directory. Use the generated heldout file for the later checks.

Collection prints an export directory and readiness flags. A three-question smoke
can produce zero training or validation rows; do not change the split to manufacture
readiness. After checking traces and coverage, `SAMPLES=-1` collects the complete
training source. Set `DISTILL_RUN_DIR` to resume an existing compatible run; manifests
protect against changed settings. Each candidate receives its own deterministic seed.

For multiple rounds, build a fresh export using the original student tokenizer:

```bash
python scripts/build_explicit_distillation_sft.py /absolute/path/to/round1/collection /absolute/path/to/round2/collection --data "$TRAIN_DATA" --heldout "$HELDOUT_DATA" --tokenizer "$SFT_STUDENT_MODEL_PATH" --output "$SCRATCH/contextgraph_sft/explicit-merged" --max-length 65536 --require-complete
```

Then test the training path with the export containing both train and validation:

```bash
EXPLICIT_SFT_DIR="$SCRATCH/contextgraph_sft/explicit-merged" TOTAL_TRAINING_STEPS=1 bash scripts/train_explicit_distillation_sft_4node_idev.sh
```

For a full run, provide `TOTAL_TRAINING_STEPS` and enough `TOTAL_EPOCHS` for that data
and global batch. Do not copy the screenshot's 124 steps without recomputing the
number of batches. This launcher does not automatically launch final evaluation.
Evaluation must use the same explicit thinking JSON interface and student tokenizer,
without collection guidance or framework graph maintenance. Auto/controller scores
answer a different question; keep the protocols and 64K budget labelled separately.
