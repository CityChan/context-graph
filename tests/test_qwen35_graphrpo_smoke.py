import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
pytestmark = pytest.mark.skipif(not BASH, reason='Bash unavailable')


@pytest.mark.parametrize('outcome,code', [
    ('complete', 0), ('missing_shard', 1), ('trainer_failure', 17),
    ('credit_failure', 19), ('missing_rollout', 1),
])
@pytest.mark.parametrize('backend', ['old_policy_counterfactual_qa', 'evidence'])
def test_five_node_two_step_smoke_requires_checkpoint_and_credit(tmp_path, outcome, code, backend):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'data').mkdir()
    for name in ('bc_train.parquet', 'bc_test.parquet'):
        (tmp_path / 'data' / name).write_text('fixture')
    (tmp_path / 'scripts/check_qwen3_observation_tokens.sh').write_text('exit 0\n')
    (tmp_path / 'scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh').write_text('''set -eu
env > "$PROJECT_ROOT/captured.env"
printf '%s\\n' "$@" > "$PROJECT_ROOT/captured.args"
if [ "$TEST_OUTCOME" = trainer_failure ]; then exit 17; fi
mkdir -p "$CHECKPOINT_ROOT/global_step_2/actor" "$ROLLOUT_DATA_DIR"
printf 2 > "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
for rank in 0 1 2 3; do
  for kind in model optim extra_state; do
    if [ "$TEST_OUTCOME:$rank:$kind" = missing_shard:3:optim ]; then continue; fi
    printf fixture > "$CHECKPOINT_ROOT/global_step_2/actor/${kind}_world_size_4_rank_${rank}.pt"
  done
done
printf '{}\\n' > "$ROLLOUT_DATA_DIR/1.jsonl"
if [ "$TEST_OUTCOME" != missing_rollout ]; then printf '{}\\n' > "$ROLLOUT_DATA_DIR/2.jsonl"; fi
''', encoding='utf8', newline='\n')
    setup = tmp_path / 'run.sh'
    setup.write_text('\n'.join([
        'scontrol() { printf "%s\\n" node0 node1 node2 node3 node4; }',
        'git() { if [ "$1" = rev-parse ]; then echo fake-sha; fi; return 0; }',
        'source() { :; }', 'conda() { :; }', 'find() { echo /dev/null; }',
        'python() { unset LD_PRELOAD; case "$1" in scripts/audit_*) printf "%s\\n" "$*" >> "$PROJECT_ROOT/audits"; if [ "$TEST_OUTCOME" = credit_failure ] && [[ "$1" = scripts/audit_*graph_credit.py ]]; then return 19; fi ;; esac; return 0; }',
        'export -f scontrol git source conda find python',
        f'bash "{(ROOT / "scripts/train_bcp_qwen35_9b_50step.sh").as_posix()}" contextgraph',
    ]) + '\n', encoding='utf8', newline='\n')
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               CONDA_PREFIX=tmp_path.as_posix(), SLURM_JOB_ID='fixture', SLURM_JOB_NODELIST='fixture',
               OPENAI_API_KEY='fixture', BCP_TRAIN_PROFILE='contextgraph_32k_4x4_batch',
               BCP_TRAIN_TOPOLOGY='full', BCP_GRAPH_RPO_SMOKE='1', SMOKE_TEST='0',
               PREFLIGHT_ONLY='0', TEST_OUTCOME=outcome, GRAPH_RPO_CREDIT_BACKEND=backend)
    result = subprocess.run([BASH, setup.as_posix()], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == code, result.stdout + result.stderr
    run_dir = next((tmp_path / 'outputs').iterdir())
    assert (run_dir / 'smoke-complete.txt').exists() == (outcome == 'complete')
    captured = dict(line.split('=', 1) for line in (tmp_path / 'captured.env').read_text().splitlines() if '=' in line)
    for key, value in {
        'TOTAL_TRAINING_STEPS': '2', 'TRAIN_BATCH_SIZE': '4', 'ROLLOUT_N': '4',
        'PPO_MINI_BATCH_SIZE': '4', 'EXPECTED_NUM_NODES': '5', 'CONTEXT_LENGTH': '32768',
        'RESPONSE_LENGTH': '24576', 'VAL_BEFORE_TRAIN': 'False', 'SAVE_FREQ': '2',
        'ADV_ESTIMATOR': 'graphrpo', 'POLICY_LOSS_MODE': 'graphrpo',
        'USE_KL_LOSS': 'True', 'BC_CONTROLLER_ACTION_POLICY': 'balanced' if backend == 'evidence' else 'structural',
    }.items():
        assert captured[key] == value
    if outcome == 'complete':
        audits = (tmp_path / 'audits').read_text()
        assert 'audit_bc_judge_results.py' in audits
        if backend == 'evidence':
            assert 'audit_evidence_graph_credit.py' in audits and '--require-nonzero' in audits
        else:
            assert 'audit_counterfactual_graph_credit.py' in audits
            assert '--fail-on-integrity-error --min-tag-rate 0.9 --require-nonzero-delta' in audits
    elif outcome == 'credit_failure':
        assert 'stage=graph_credit_audit exit=19' in (run_dir / 'failure.log').read_text()


@pytest.mark.parametrize('method,profile', [
    ('foldagent', 'foldagent_32k_small_batch'), ('contextgraph', 'default'),
])
def test_graph_smoke_rejects_incompatible_method_or_profile(tmp_path, method, profile):
    result = subprocess.run([BASH, (ROOT / 'scripts/train_bcp_qwen35_9b_50step.sh').as_posix(), method],
                            cwd=tmp_path, env=dict(os.environ, BCP_GRAPH_RPO_SMOKE='1',
                                                  BCP_TRAIN_PROFILE=profile, SMOKE_TEST='0'),
                            capture_output=True, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 2
    assert 'GraphRPO smoke requires' in result.stderr
