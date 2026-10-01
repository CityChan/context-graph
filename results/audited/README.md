# Audited saved evaluations

These scores were recalculated from saved per-instance records and checked against all three shard manifests and summaries. This is an artifact-consistency audit, not a fresh evaluation or independent verification of the judge's answers.

Full provenance, source SHA-256 hashes, selected indices, protocol settings and answer-free per-instance outcomes are in [evaluations.json](evaluations.json). Raw prompts, answers, credentials and local paths are not published.

GAIA uses the local BrowseComp-Plus corpus, not official GAIA retrieval. Small BC-P subsets are diagnostics. Runs span different commits and settings; compare protocols before comparing scores. Historical finalizer behavior is preserved in these results; the October 1 fixes do not retroactively improve them. Missing checkpoint revisions remain unknown. These records do not contain SWE-bench Verified scores.

| Run | Benchmark | Model | Method | Correct / evaluated | Accuracy | Context | Commit |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `bcp-qwen35-9b-contextgraph-1028991-20260927_192240` | bcp | Qwen/Qwen3.5-9B | contextgraph | 2/8 | 25.00% | 32768 | `0399225` |
| `bcp-qwen35-9b-contextgraph-1028991-20260927_200057` | bcp | Qwen/Qwen3.5-9B | contextgraph | 74/150 | 49.33% | 32768 | `0399225` |
| `bcp-qwen35-9b-foldagent-1029161-20260927_192246` | bcp | Qwen/Qwen3.5-9B | foldagent | 2/8 | 25.00% | 32768 | `0399225` |
| `bcp-qwen35-9b-foldagent-1029161-20260927_200103` | bcp | Qwen/Qwen3.5-9B | foldagent | 68/150 | 45.33% | 32768 | `0399225` |
| `bcp-qwen35-9b-react-1030584-20260928_033036` | bcp | Qwen/Qwen3.5-9B | react | 65/150 | 43.33% | 32768 | `dbdae74` |
| `gaia-local-qwen35-9b-contextgraph-1028991-20260927_214457` | gaia | Qwen/Qwen3.5-9B | contextgraph | 9/127 | 7.09% | 32768 | `8ca0dd0` |
| `gaia-local-qwen35-9b-foldagent-1029161-20260927_214502` | gaia | Qwen/Qwen3.5-9B | foldagent | 12/127 | 9.45% | 32768 | `8ca0dd0` |
| `gaia-local-qwen35-9b-react-1029161-20260928_002508` | gaia | Qwen/Qwen3.5-9B | react | 9/127 | 7.09% | 32768 | `6883854` |

## Excluded runs

Exclusions are retained to make the selection visible; they are not scored as clean evaluations.

- `bcp-qwen38-contextgraph-1028991-20260927_150623`: Required evidence field missing: model.
- `bcp-qwen38-contextgraph-1028991-20260927_153143`: Required evidence field missing: model.
- `bcp-qwen38-contextgraph-1029161-20260927_145717`: Execution or judge parse failures; not a clean score.
- `bcp-qwen38-foldagent-1028991-20260927_145712`: Missing, duplicated, or misplaced evaluation rows.
- `bcp-qwen38-foldagent-1029161-20260927_150613`: Required evidence field missing: model.
- `bcp-qwen38-foldagent-1029161-20260927_153149`: Required evidence field missing: model.

## SWE-bench Verified: ARM compatibility pilot

This uses the **Verified dataset**, but is **not an official x86 SWE-bench score**. The dataset catalog has 500 tasks; only one task was selected. Generation's saved `grading_status=pending` predates the separate completed ARM grading summary.

| Run | Method | Instance | Resolved | Patch | Context |
| --- | --- | --- | ---: | --- | ---: |
| `swe-foldagent-1035199-hgEsOA` | foldagent | `sympy__sympy-20590` | 0/1 | Empty | 32768 |

Prediction SHA-256, empty patch bytes, instance IDs and manifest/grading metadata agree in the synced files. No local calibration report or image/dataset payload is available for revalidation. Later 64K Lite console transcripts are not included as artifact-audited results.

## Reproduce from the original local artifacts

```bash
python scripts/export_audited_results.py --source outputs --destination results/audited --arm-source logs/swe-foldagent-1035199-hgEsOA
```

The original files are required to reproduce their hashes. The published answer-free rows suffice to recalculate the reported accuracy; they do not suffice to rerun grading.
