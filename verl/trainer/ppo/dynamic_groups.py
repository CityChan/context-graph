"""Bounded, whole-group rollout selection before any policy update."""
from collections import OrderedDict

import numpy as np
import torch


def informative_groups(batch, algorithm, role=None):
    """Return row indices per useful uid, keeping all streams of every candidate.

    M credit already contains same-state LOO advantages. E uses outcomes only.
    Ordinary GraphRPO can learn from a negative process label even on a tie.
    FoldGRPO deliberately filters on deduplicated outcome variance only.
    """
    if algorithm.get('graphrpo_memory_only', False):
        role = 'M'
    if 'rm_scores' not in batch.batch:
        raise ValueError('Dynamic sampling requires rollout task rewards (rm_scores)')
    groups = OrderedDict()
    for i, uid in enumerate(batch.non_tensor_batch['uid']):
        groups.setdefault(str(uid), []).append(i)
    rewards = batch.batch['rm_scores'].sum(-1)
    mask = batch.batch['response_mask'].bool().clone()
    if 'mask_rollout' in batch.batch:
        mask &= ~batch.batch['mask_rollout'].bool().reshape(-1, 1)
    keep = []
    for rows in groups.values():
        if not mask[rows].any():
            continue
        audits = batch.non_tensor_batch.get('graph_rpo_continuation')
        if audits is not None and any((audits[i] or {}).get('skipped') for i in rows):
            continue
        outcomes = {}
        for i in rows:
            key = str(batch.non_tensor_batch['gen_uid'][i])
            value = float(rewards[i])
            if not np.isfinite(value):
                raise ValueError('Nonfinite rollout reward in dynamic sampling')
            if key in outcomes and outcomes[key] != value:
                raise ValueError('Streams disagree on candidate reward')
            outcomes[key] = value
        if len(outcomes) < 2:
            continue
        useful = len(set(outcomes.values())) > 1
        if role == 'M':
            credit = batch.batch['graph_edit_credit_mask'][rows]
            if not torch.isfinite(credit[mask[rows]]).all():
                raise ValueError('Nonfinite active maintenance credit in dynamic sampling')
            useful = bool((torch.isfinite(credit) & (credit != 0) & mask[rows]).any())
        elif role is None and str(algorithm.adv_estimator).lower() == 'graphrpo':
            for key, coefficient, negative in [('process_reward_mask', 'graphrpo_beta', True),
                                                ('graph_edit_credit_mask', 'graphrpo_alpha', False)]:
                if key in batch.batch and float(algorithm.get(coefficient, 1)) > 0:
                    labels = batch.batch[key][rows]
                    if not torch.isfinite(labels[mask[rows]]).all():
                        raise ValueError('Nonfinite active process/graph credit in dynamic sampling')
                    signal = labels < 0 if negative else labels != 0
                    useful |= bool((torch.isfinite(labels) & signal & mask[rows]).any())
        if useful:
            keep.append(rows)
    return keep, len(groups)


def sample_groups(first, following, generate, algorithm, role, target, metrics):
    """Consume at most max_gen_batches from the SAME epoch iterator."""
    from verl import DataProto

    prefix = str(algorithm.adv_estimator).lower()
    if not algorithm.get(prefix + '_dynamic_sampling', False):
        return generate(first)
    limit = int(algorithm.get(prefix + '_dynamic_max_gen_batches', 4))
    minimum = int(algorithm.get(prefix + '_dynamic_min_groups', 1))
    if not 1 <= minimum <= target or limit < 1:
        raise ValueError('Dynamic sampling requires 1 <= min_groups <= train_batch_size and max_gen_batches >= 1')
    selected, seen, batches = [], 0, 0
    current = first
    while True:
        batch = generate(current)
        batches += 1
        groups, total = informative_groups(batch, algorithm, role)
        seen += total
        for rows in groups[:target - len(selected)]:
            selected.append(batch[np.asarray(rows)])
        if len(selected) >= minimum or batches >= limit:
            break
        try:
            current = next(following)
        except StopIteration:
            break
    metrics.update({prefix + '/dynamic_gen_batches': batches,
                    prefix + '/dynamic_seen_groups': seen,
                    prefix + '/dynamic_kept_groups': len(selected),
                    prefix + '/dynamic_min_groups_met': int(len(selected) >= minimum)})
    if not selected:
        batch.batch['response_mask'].zero_()
        batch.meta_info['dynamic_empty'] = True
        return batch
    return DataProto.concat(selected)
