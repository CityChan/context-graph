# Scripts

This directory keeps runnable entry points for the active benchmark tracks.

## BrowseComp-Plus

- `eval_bc.py`: direct evaluation entry point.
- `eval_bc_*.sh`: zero-shot evaluation wrappers.
- `train_bc_*.sh`: training wrappers for ContextGraph, FoldAgent, and ReAct/baseline variants.
- `smoke_train_bc_*.sh`: short training smoke runs.
- `smoke_train_bc_pair_8b_5node_1step_32k_active.sh`: runs the FoldAgent and
  ContextGraph production one-step smokes serially in one five-node idev.

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

## Removed Surface

HotpotQA, MuSiQue, and 2WikiMultiHopQA wrappers were removed from the active script surface. Shared search code remains for BrowseComp-Plus.
