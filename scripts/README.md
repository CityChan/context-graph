# Scripts

This directory keeps runnable entry points for the active benchmark tracks.

## Current Qwen3.5-9B workflows

- `train_bcp_qwen35_9b_contextgraph_32k_4x4_5node.sbatch`: five nodes,
  one search + four trainers, 32K, batch 4 x rollout 4, 50 updates.
- `train_bcp_qwen35_9b_50step.sh`: shared profile implementation; wrappers
  carry method/topology/batch choices and are intentionally retained.
- `smoke_bcp_qwen35_9b_4node_idev.sh`: short distributed update smoke.
- `eval_bcp_qwen38.py`: shared token-ID evaluator despite the historical filename;
  supports the current BC-P and local-GAIA Qwen3.5 launchers.
- `evaluation_records.py`: shared shard/provenance/row validation.
- `export_audited_results.py`: reproducible, sanitized publication of saved evidence.

See [RL setup and validation limits](../docs/bcp_qwen35_9b_rl.md) and
[published results](../results/README.md). Do not interpret a launcher or
dependency preflight as successful GPU training.

## WideSearch and ScienceWorld

For WideSearch and ScienceWorld, use `eval_agent_benchmarks_qwen35_9b_4node.sbatch`
with a benchmark and `both`, `contextgraph`, or `foldagent`. It prepares data,
starts inference, runs resumable per-task evaluation and saves trajectories.
See [protocol and runnable commands](../docs/widesearch_scienceworld_eval.md).
The underlying tools are `prepare_agent_benchmarks.py`, `eval_agent_benchmarks.py`
and the separate `grade_widesearch.py` official-metric wrapper.

## SWE-bench Verified and Lite

- `eval_swebench_verified.py`: pinned dataset preparation, patch generation and audits.
- `run_swe_lite_docker.sh`: Lite generation and official-harness grading on x86 Docker.
- `run_swe_arm_pilot_idev.sh`: one supported SymPy task on Vista ARM/Apptainer;
  requires a compute allocation, including for calibration.
- `grade_swe_arm_pilot.py`: separate ARM calibration and grading.
- `serve_swe_qwen35_9b_vista.sbatch`: model serving on a GH compute node.

The [SWE guide](../docs/swebench_verified_eval.md) explains runtime boundaries.
The ARM pilot is not a full Verified/Lite result.

## BrowseComp-Plus

- `eval_bc.py`: direct evaluation entry point.
- `eval_bc_*.sh`: zero-shot evaluation wrappers.
- `train_bc_*.sh`: training wrappers for ContextGraph, FoldAgent, and ReAct/baseline variants.
- `smoke_train_bc_*.sh`: short training smoke runs.
- `smoke_train_bc_pair_8b_5node_1step_32k_active.sh`: runs the FoldAgent and
  ContextGraph production one-step smokes serially in one five-node idev.
- `smoke_train_bc_pair_8b_4node_1step_32k_active.sh`: four-node idev variant
  using one search node and three trainer ranks.

BrowseComp-Plus uses `envs/search_server.py` and `LOCAL_SEARCH_URL`.

## GAIA

- `make_gaia_data.py`: builds GAIA ReAct/FoldAgent/ContextGraph parquets.
- `eval_gaia.py`: API-based evaluator for small GAIA validation subsets.
- `eval_gaia_graph_api_smoke.sh`: starts the BrowseComp search server locally,
  probes it, then runs a small GAIA ContextGraph API smoke.

The initial GAIA integration targets text-only rows and uses the same search
tools as BrowseComp-Plus. Rows with file attachments are skipped by default.

## ScienceAgentBench

- `make_sab_data.py`: builds SAB parquets from the upstream CSV and benchmark package.
- `train_sab.py`: SAB training/evaluation entry point.
- `eval_sab_*.sh`: SAB eval wrappers.
- `scan_sab_missing_imports.py`: helper for checking package coverage.

SAB uses `envs/scienceagent_env.py` and `envs/scienceagent_sandbox.py`.

## ALFWorld

- `make_alfworld_data.py`: builds real/hard ALFWorld parquets.
- `train_alfworld_*.sh`: training wrappers.
- `test_alfworld_*.sh` and `smoke_alfworld_*.sh`: smoke/diagnostic runs.
- `val_alfworld_*.sh`: checkpoint validation wrappers.

ALFWorld uses `envs/alfworld_env.py` and does not need a search server.

## SkillRL Search-R1 reference diagnostic

- `prepare_skillrl_search_reference_vista_batch.sh`: one-node Vista batch entry
  point for all setup, downloads, index assembly, corpus decompression, and data
  sampling. It calls `prepare_skillrl_search_reference_vista.sh`, creates a
  separate `skillrl_search` environment, and never modifies the formal
  `cxtgraph` environment.
- `run_skillrl_search_reference_2node_idev.sh`: uses one GH200 node for the
  official dense retriever and one GH200 node for evaluation. It compares
  Qwen2.5-7B-Instruct and the released Search SFT checkpoint on all seven
  benchmarks; the optional released RL checkpoint adds a third rung. Set
  `EVAL_TARGET=qwen`, `sft`, or `rl` to run only one rung; the default is `all`.
- `sample_searchr1_reference_data.py` and
  `audit_skillrl_search_reference.py`: deterministic sampling and log auditing.

This is an intentionally independent reference path for diagnosing the local RL
implementation. The two-node script is evaluation-only. A paper-scale GRPO
update on Vista requires a separate allocation with one retriever node and four
one-GH200 trainer nodes.

## Multi-turn ContextGraph SFT

- `eval_interactive.py`: shared API trajectory runner for ALFWorld and ScienceWorld.
  Graph workflows default to controller-owned action checkpoints: the controller
  freezes a bounded legal-node snapshot, vLLM constrains the teacher to a JSON
  schema over legal graph actions and candidate indices, and normal environment turns do
  not expose graph XML tools. Use `--no-structured-graph-controller` only for a
  legacy-protocol comparison.
- `build_contextgraph_sft.py`: validates structured graph traces, filters failed or
  redundant graph control, groups tasks across splits, and writes multi-turn SFT Parquet.
- `smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh`: performs one real
  Qwen3.6-27B multi-turn SFT optimizer step on four GH200 nodes and verifies the
  four checkpoint shards. It defaults to LoRA rank 32, PyTorch SDPA, and 4-way
  FSDP2 data parallelism. The single accepted trajectory is repeated once per
  rank only for this optimizer-path smoke.
- `../requirements_qwen36_sft.txt`: minimal pure-Python training additions for
  the existing `deepseek_v4` inference environment; the SDPA smoke deliberately
  does not install or import the external `flash-attn` extension.
- `smoke_interactive_ctxgraph_qwen36_27b_1node.sh`: two-domain Qwen3.6-27B smoke.
- `generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh`: two-domain DeepSeek-V4
  production teacher job; array 0 is ALFWorld and array 1 is ScienceWorld.
- `smoke_interactive_ctxgraph_deepseek_v4_4node_idev.sh`: conservative TP=4,
  32K-context DeepSeek-V4 smoke for an existing four-GH200 allocation.

## Removed Surface

Older standalone HotpotQA, MuSiQue and 2WikiMultiHopQA wrappers were removed.
The separately maintained Search-R1 reference diagnostics above still include
multi-hop datasets. Shared search code remains for BrowseComp-Plus.
