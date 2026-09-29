"""Check the selected logging SDK without creating a run or contacting W&B."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def check_logging(env=None):
    env = os.environ if env is None else env
    disabled = env.get('BC_DISABLE_WANDB') == '1'
    required = env.get('BC_REQUIRE_WANDB') == '1'
    if required and disabled:
        raise RuntimeError('BC_REQUIRE_WANDB=1 conflicts with BC_DISABLE_WANDB=1')
    if required and not env.get('WANDB_API_KEY'):
        raise RuntimeError('BC_REQUIRE_WANDB=1 but WANDB_API_KEY is not set')
    if disabled or not env.get('WANDB_API_KEY'):
        return {'backend': 'console', 'python': sys.executable}
    module = importlib.import_module('wandb')
    origin = getattr(module, '__file__', None)
    paths = list(getattr(module, '__path__', []))
    missing = [name for name in ('init', 'log', 'finish', 'Settings')
               if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            f'Invalid W&B SDK: missing callable {missing}; file={origin!r}; paths={paths!r}; '
            f'python={sys.executable}. Check this interpreter\'s wandb installation and import shadowing. '
            'A repository wandb/ log directory can import as a namespace when the SDK is absent. '
            'Do not delete experiment logs to repair the SDK.')
    try:
        version = importlib.metadata.version('wandb')
    except importlib.metadata.PackageNotFoundError:
        version = 'unknown'
    return {'backend': 'console,wandb', 'python': sys.executable, 'file': origin,
            'version': version, 'api_verified': True, 'network_verified': False}


if __name__ == '__main__':
    try:
        report = check_logging()
    except Exception as exc:
        report = {'error': f'{type(exc).__name__}: {exc}', 'python': sys.executable}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(1 if 'error' in report else 0)
