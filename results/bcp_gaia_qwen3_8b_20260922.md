# BC-P and GAIA Qwen3-8B evaluation record

Recorded: 2026-09-22

This file separates matched finalizer controls from later StructMem runs. Do
not combine rows across protocol groups as a single controlled comparison.

## BC-P full test split (150 examples)

### Matched finalizer controls

All three runs use Qwen3-8B, 64K context, four nodes, 100 maximum turns, and a
1024-token final-answer reserve.

| Method | Correct | Accuracy | Source log |
|---|---:|---:|---|
| ReAct + Finalizer | 43/150 | 28.67% | `logs/eval-bc-8b-baseline-64k-finalizer.916033.out` |
| FoldAgent + Finalizer | 56/150 | 37.33% | `logs/eval-bc-8b-foldagent-64k-finalizer.916034.out` |
| ContextGraph + Finalizer | 48/150 | 32.00% | `logs/eval-bc-8b-contextgraph-64k-finalizer.916035.out` |

### StructMem runs

| Method | Correct | Accuracy | Status | Source |
|---|---:|---:|---|---|
| ContextGraph + StructMem + Finalizer (pre-fix) | 39/150 | 26.00% | Complete, but averaged 0.5533 extraction errors per example | `logs/eval-bc-gaia-base8b-structmem-highbudget-4n.1010384.20260920_232150.log` |
| ContextGraph + StructMem + Finalizer (current fixed) | pending | pending | Rerun entrypoint committed as `scripts/submit_eval_bcp_structmem_finalizer_qwen3_8b_4node.sh` | explicit tee log will be `logs/bcp-smfin8b.<jobid>.<timestamp>.log` |

An additional later ContextGraph + Finalizer run without StructMem scored
38/150 (25.33%) in `logs/eval-bc-gaia-base-idev.1008275.log`. It is not part
of the matched 916033--916035 control triplet.

## GAIA full local split (127 examples)

### Matched finalizer controls

All three runs use Qwen3-8B, 32K working context, and a 1024-token
final-answer reserve.

| Method | Correct | Accuracy | Source log |
|---|---:|---:|---|
| ReAct + Finalizer | 5/127 | 3.94% | `logs/gaia-baseline-8b-32k-zeroshot.916343.out` |
| FoldAgent + Finalizer | 16/127 | 12.60% | `logs/gaia-foldagent-8b-32k-zeroshot.916344.out` |
| ContextGraph + Finalizer | 15/127 | 11.81% | `logs/gaia-ctxgraph-8b-32k-zeroshot.916345.out` |

### StructMem runs

| Method | Correct | Accuracy | Finish rate | Status | Source |
|---|---:|---:|---:|---|---|
| ContextGraph + StructMem + Finalizer (pre-fix) | 10/127 | 7.87% | 100% | Complete, with extraction/gap errors | `logs/eval-bc-gaia-base8b-structmem-highbudget-4n.1010384.20260920_232150.log` |
| ContextGraph + StructMem + Finalizer (current fixed) | 7/127 | 5.51% | 100% | Complete; one extraction error and one controller error in total | Captured from the completed direct-idev terminal output; the main local log was not preserved |

An additional later ContextGraph + Finalizer run without StructMem scored
8/127 (6.30%) in `logs/eval-bc-gaia-base-idev.1008275.log`. It is not part of
the matched 916343--916345 control triplet.

## Interpretation constraints

- The matched control rows and StructMem rows were produced at different code
  snapshots. They are historical evidence, not a final matched four-way table.
- The current fixed BC-P treatment must complete before the fixed StructMem
  result can be compared with newly rerun controls.
- The local GAIA setup uses the BrowseComp-Plus corpus rather than live web
  retrieval. Treat its scores as internal comparisons, not official GAIA or
  paper-comparable scores.
- ReAct/FoldAgent comparisons measure the complete agent designs. The isolated
  StructMem effect is measured only by comparing ContextGraph + Finalizer
  against ContextGraph + StructMem + Finalizer under the same commit and
  evaluation protocol.
