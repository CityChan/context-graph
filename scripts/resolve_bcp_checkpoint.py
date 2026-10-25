#!/usr/bin/env python3
"""Resolve an offline checkpoint without requiring HF refs/main or downloading."""
import json
import os
from pathlib import Path
import sys


def validate(path, scratch, model_id):
    path = Path(path).resolve()
    if not path.is_relative_to(scratch):
        raise ValueError('Model must be under SCRATCH')
    expected = 'models--' + model_id.replace('/', '--')
    for parent in path.parents:
        if parent.name.startswith('models--') and parent.name != expected:
            raise ValueError(f'MODEL_PATH belongs to {parent.name}, not {model_id}')
    try:
        json.loads((path / 'config.json').read_text())
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        shards = set(index['weight_map'].values())
        if not shards or not all((path / s).is_file() and (path / s).stat().st_size > 0 for s in shards):
            raise ValueError('missing or empty weight shards')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Incomplete checkpoint {path}: {exc}') from exc
    return path


def resolve(scratch, model_id, explicit=''):
    scratch = Path(scratch).resolve()
    if explicit:
        return validate(explicit, scratch, model_id)
    cache = scratch / 'hf_cache' / 'hub'
    repo = cache / ('models--' + model_id.replace('/', '--'))
    ref = repo / 'refs' / 'main'
    if ref.is_file():
        # Honor the cached revision; do not silently substitute older weights.
        return validate(repo / 'snapshots' / ref.read_text().strip(), scratch, model_id)
    complete = []
    for path in sorted((repo / 'snapshots').glob('*')):
        try:
            complete.append(validate(path, scratch, model_id))
        except ValueError:
            pass
    if len(complete) == 1:
        print(f'No refs/main; using the only complete {model_id} snapshot: {complete[0]}', file=sys.stderr)
        return complete[0]
    if complete:
        raise ValueError('Multiple complete snapshots; set MODEL_PATH explicitly:\n' + '\n'.join(map(str, complete)))
    cached = sorted(p.name.removeprefix('models--').replace('--', '/') for p in cache.glob('models--*') if p.is_dir())
    raise ValueError(f'No complete local checkpoint for {model_id} under {repo}.\n'
                     f'Cached repository directories (not proof of complete weights): {", ".join(cached) or "none"}\n'
                     'Set MODEL_ID to the intended cached model, or set MODEL_PATH to its complete checkpoint. No download attempted.')


if __name__ == '__main__':
    try:
        print(resolve(os.environ['SCRATCH'], os.environ['MODEL_ID'], os.environ.get('MODEL_PATH', '')))
    except (ValueError, OSError) as exc:
        print(f'CHECKPOINT_PREFLIGHT_FAILED: {exc}', file=sys.stderr)
        sys.exit(2)
