# Results

## Locally audited artifacts

[Audited evaluation bundle](audited/README.md): eight complete BC-P/local-GAIA
runs, six explicitly excluded runs, and one empty-patch SWE-bench Verified ARM
pilot. The JSON includes provenance hashes and answer-free per-instance outcomes.
Scores are historical and retain the protocol at their recorded commit.

Here, **audited** means consistency between local saved artifacts. It does not
mean independently rejudged answers, official benchmark certification, or a
new run of the current code. **SWE-bench Verified** is a dataset name; the saved
ARM result covers one of its tasks and is not an official x86 score.

## Older recorded summaries

- [Qwen3-8B BC-P and GAIA, September 22](bcp_gaia_qwen3_8b_20260922.md)
- [Qwen3-8B BC-P 64K, August 9](browsecomp_plus_8b_64k_zeroshot_2026-08-09.md)
- [Qwen3-8B ReAct BC-P 32K, August 15](browsecomp_plus_8b_react_32k_zeroshot_2026-08-15.md)

These historical summaries are retained with their original qualifications.
They were not upgraded to the new artifact-audit tier during this review.
Different sample sets, budgets, judge behavior, finalizers, memory modes and
commits must not be combined into a single controlled leaderboard.
