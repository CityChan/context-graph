import json

import pytest

from scripts.evaluation_resume import prepare_resume, preserve_interrupted_artifacts


def test_resume_validates_protocol_and_recovers_only_torn_last_write(tmp_path):
    manifest = {'indices':[0,1,2,3], 'commit':'sha', 'temperature':0}
    assert prepare_resume(tmp_path,0,manifest) == set()
    path = tmp_path / 'results-0.jsonl'
    path.write_text('{"source_index":0,"status":"ok"}\n{"source_index":', encoding='utf8')
    assert prepare_resume(tmp_path,0,manifest,True) == {0}
    assert path.read_text().endswith('\n')
    with pytest.raises(ValueError, match='mismatch'):
        prepare_resume(tmp_path,0,{**manifest,'temperature':.5},True)
    with pytest.raises(FileExistsError):
        prepare_resume(tmp_path,0,manifest)
    path.write_text('bad\n{"source_index":3}\n')
    with pytest.raises(ValueError):
        prepare_resume(tmp_path,0,manifest,True)


def test_resume_preserves_incomplete_audits_and_rejects_duplicate_tasks(tmp_path):
    manifest = {'indices':[0,1,2,3]}
    prepare_resume(tmp_path,0,manifest)
    path = tmp_path/'results-0.jsonl'
    path.write_text('{"source_index":0}\n{"source_index":0}\n')
    with pytest.raises(ValueError, match='duplicate'):
        prepare_resume(tmp_path,0,manifest,True)
    request = tmp_path/'requests-3.jsonl'
    request.write_text('previous attempt')
    preserve_interrupted_artifacts(tmp_path,3)
    assert not request.exists()
    assert next((tmp_path/'interrupted').glob('*/requests-3.jsonl')).read_text() == 'previous attempt'
