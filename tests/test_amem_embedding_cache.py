"""Regression for strict Hub completeness checks on an intentionally filtered cache."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agents import graph_memory_baselines as memory
from scripts import prepare_amem_embedding as probe


@pytest.fixture
def cache(monkeypatch, tmp_path):
    requests, loads = [], []

    def download(repo, *, revision, local_files_only, allow_patterns=None):
        requests.append(dict(repo=repo, revision=revision, local_files_only=local_files_only,
                             allow_patterns=allow_patterns))
        # Match the reported newer-Hub behavior: whole-repo offline requests fail
        # even though every file needed by the PyTorch loader is cached.
        if allow_patterns != ['*.json', '*.txt', '*.safetensors']:
            raise RuntimeError('IncompleteSnapshotError: missing README.md, onnx/model.onnx')
        assert repo == memory.EMBEDDING_MODEL and revision == memory.EMBEDDING_REVISION
        return str(tmp_path)

    class Model:
        def __init__(self, path, *, device, local_files_only):
            loads.append((path, device, local_files_only))
        def encode(self, text, **kwargs):
            return np.array([1., 0.]) if 'Paris' in text else np.array([0., 1.])

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Model, __version__='fixture'))
    memory.embedding_model.cache_clear()
    yield requests, loads
    memory.embedding_model.cache_clear()


@pytest.mark.parametrize('offline', [False, True])
def test_preparation_and_runtime_share_filtered_cache(cache, monkeypatch, capsys, offline):
    monkeypatch.setattr(sys, 'argv', ['prepare_amem_embedding'] + (['--offline'] if offline else []))
    probe.main()
    requests, loads = cache
    assert len(requests) == 2
    assert requests[0]['local_files_only'] is offline
    assert requests[1] == {**requests[0], 'local_files_only': True}
    assert len(loads) == 1 and loads[0][1:] == ('cpu', True)
    report = json.loads(capsys.readouterr().out)
    assert report['event'] == 'AMEM_EMBEDDING_OK' and report['semantic_retrieval_passed']


def test_runtime_missing_required_files_fails_without_network_fallback(cache, monkeypatch):
    requests = []
    def missing(*args, **kwargs):
        requests.append(kwargs)
        raise RuntimeError('missing model.safetensors')
    monkeypatch.setattr(sys.modules['huggingface_hub'], 'snapshot_download', missing)
    with pytest.raises(RuntimeError, match='missing model.safetensors'):
        memory.embed('test')
    assert len(requests) == 1 and requests[0]['local_files_only'] is True
    assert cache[1] == []
