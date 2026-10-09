"""Select saved AgentFold format failures without reindexing the original data."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation


def prepare(source, destination):
    from omegaconf import OmegaConf
    from scripts.eval_bcp_qwen38 import config_for

    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifests, rows, _ = read_evaluation(source)
    manifest = manifests[0]
    if manifest['method'] != 'agentfold' or manifest.get('benchmark', 'bcp') != 'bcp':
        raise ValueError('Expected a completed BC-P AgentFold run')
    if manifest['model'] != 'Qwen/Qwen3.5-9B':
        raise ValueError('This launcher supports Qwen3.5-9B only')
    current = config_for(SimpleNamespace(method='agentfold', memory_mode='repaired'))
    if manifest['config'] != OmegaConf.to_container(current):
        raise ValueError('Source config differs from the supported repaired protocol')
    data = Path(manifest['data_path'])
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Original dataset hash mismatch')
    indices = []
    for row in rows:
        index = row['source_index']
        trajectory = json.loads((source / f'trajectory-{index}.json').read_text(encoding='utf-8'))
        failed = trajectory.get('termination_reason') == 'invalid_tool_limit'
        if bool(row.get('env_stats', {}).get('hit_format_retry_limit')) != failed:
            raise ValueError(f'Result/trajectory format-failure mismatch for {index}')
        if failed:
            indices.append(index)
    if not indices:
        raise ValueError('No invalid_tool_limit tasks to retry')
    destination.mkdir(parents=True, exist_ok=True)
    selection = destination / 'indices.json'
    plan = dict(scope='failure-selected diagnostic subset; do not merge with the original benchmark',
                source_run=str(source), source_commit=manifest['commit'],
                source_sha256=manifest['source_sha256'], indices=indices, count=len(indices),
                original_size=len(rows), selection_reason='invalid_tool_limit')
    for path, value in [(selection, indices), (destination / 'retry-plan.json', plan)]:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2)
    rollout = manifest['config']['actor_rollout_ref']['rollout']
    variables = dict(DATA_PATH=str(data.resolve()), EVAL_INDICES_FILE=str(selection),
                     RUN_ROOT=str(destination / 'run'), SAMPLES='-1', SEED=str(manifest['seed']),
                     MODEL_ID=manifest['model'], MODEL_PATH=manifest['model_path'],
                     JUDGE_MODEL=manifest['judge_model'], BENCHMARK='bcp', MEMORY_MODE='repaired',
                     MODEL_MAX_LEN=str(rollout['prompt_length'] + rollout['response_length']),
                     WORKERS='1', SERVER_ENFORCE_EAGER='1' if manifest.get('server_execution', {}).get(
                         'requested_enforce_eager', True) else '0')
    with (destination / 'retry-env.sh').open('x', encoding='utf-8', newline='\n') as handle:
        for key, value in variables.items():
            handle.write(f'export {key}={shlex.quote(value)}\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_run, args.destination), indent=2))
