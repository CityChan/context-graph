# Scripts

This directory keeps runnable entry points for the active benchmark tracks.

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

HotpotQA, MuSiQue, and 2WikiMultiHopQA wrappers were removed from the active script surface. Shared search code remains for BrowseComp-Plus.
