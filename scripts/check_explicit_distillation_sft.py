"""Check the exact export files, every target span, and independent heldout tasks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.exact_context import checksum, tokenizer_signature, validate_agent_context
from scripts.build_explicit_distillation_sft import PROFILE, file_hash, strip_guidance
from scripts.generate_bcp_graph_sft import question


def check(root, heldout, tokenizer, max_length):
    import pandas as pd
    from verl.utils.dataset.multiturn_sft_dataset import convert_nested_value_to_list_recursive
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    signature = tokenizer_signature(tokenizer)
    if manifest.get('profile') != PROFILE or manifest.get('tokenizer_sha256') != signature:
        raise ValueError('export_profile_or_student_tokenizer_mismatch')
    if manifest.get('heldout_sha256') != file_hash(heldout):
        raise ValueError('heldout_file_changed')
    if manifest.get('counts', {}).get('incomplete_episodes'):
        raise ValueError('collection_incomplete')
    heldout_hashes = {checksum(question(r)) for r in pd.read_parquet(heldout).to_dict('records')}
    groups, totals = {}, {}
    for split in ('train', 'validation'):
        filename = split + '.parquet'
        path = root / filename
        if manifest['files_sha256'].get(filename) != file_hash(path):
            raise ValueError('export_file_changed')
        rows = pd.read_parquet(path).to_dict('records')
        if not rows:
            raise ValueError('empty_' + split)
        groups[split] = set()
        for raw in rows:
            row = convert_nested_value_to_list_recursive(raw)
            if (row.get('sft_profile') != PROFILE or row.get('split') != split
                    or row.get('graph_action_protocol') != 'explicit_tools' or row.get('exact_replay') is not False):
                raise ValueError('row_protocol_or_split_mismatch')
            validate_agent_context(row, max_length, signature)
            _, removed = strip_guidance(row['messages'])
            if removed:
                raise ValueError('collection_guidance_remains')
            groups[split].add(row['question_hash'])
        if groups[split] & heldout_hashes:
            raise ValueError('evaluation_overlap')
        totals[split] = len(rows)
    if groups['train'] & groups['validation']:
        raise ValueError('train_validation_overlap')
    return dict(passed=True, rows=totals, tokenizer_sha256=signature,
                context=max_length, exact_replay=False, heldout_sha256=file_hash(heldout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--heldout', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--max-length', type=int, default=65536)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    result = check(args.data, args.heldout, AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True), args.max_length)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
