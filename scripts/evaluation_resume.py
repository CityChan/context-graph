"""Strict same-protocol resume for append-only three-shard evaluation."""
import json
import os
from pathlib import Path
import subprocess


def source_commit():
    root = Path(__file__).resolve().parents[1]
    if (root / '.git').exists():
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    value = os.getenv('SOURCE_COMMIT', '').strip()
    if not value:
        raise RuntimeError('Snapshot has no .git: supply SOURCE_COMMIT for provenance')
    return value


def prepare_resume(root, rank, manifest, resume=False):
    root = Path(root)
    path, results = root / f'manifest-{rank}.json', root / f'results-{rank}.jsonl'
    if path.exists():
        if not resume:
            raise FileExistsError('Existing manifest: use --resume with identical protocol')
        if json.loads(path.read_text(encoding='utf8')) != manifest:
            raise ValueError('Resume manifest mismatch; use a new output directory')
    else:
        if results.exists():
            raise ValueError('Results without provenance cannot be resumed')
        path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    completed = set()
    if results.exists():
        # Only a torn FINAL write may be discarded. Never swallow interior errors.
        content = results.read_bytes()
        lines = content.splitlines(keepends=True)
        offset = 0
        allowed = set(manifest['indices'][rank::3])
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b'\n'):
                    raise
                with results.open('r+b') as handle:
                    handle.truncate(offset)
                break
            task = row['source_index']
            if task not in allowed or task in completed:
                raise ValueError('Unexpected or duplicate task in resume results')
            completed.add(task)
            offset += len(line)
        if completed and content and not content.endswith(b'\n') and offset == len(content):
            with results.open('ab') as handle:
                handle.write(b'\n')
    else:
        results.touch(exist_ok=False)
    return completed


def preserve_interrupted_artifacts(root, index):
    """Keep unfinished attempt files while starting a clean request audit."""
    import uuid
    root = Path(root)
    files = [p for p in (root / f'requests-{index}.jsonl', root / f'trajectory-{index}.json') if p.exists()]
    if files:
        destination = root / 'interrupted' / f'{index}-{uuid.uuid4().hex}'
        destination.mkdir(parents=True)
        for path in files:
            path.rename(destination / path.name)
