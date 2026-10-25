import json
from types import SimpleNamespace

import pytest

from scripts.preflight_graph_retry_schema import run_checks, task_probe


class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.renders = []

    def apply_chat_template(self, messages, **kwargs):
        self.renders.append((messages, kwargs))
        return [10, 11, 12]

    def decode(self, ids, skip_special_tokens):
        return self.text if skip_special_tokens else self.text + '<|im_end|>'


class Client:
    def __init__(self, choice=None, error=None):
        self.completions = self
        self.choice, self.error = choice, error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[self.choice], usage=None)


def test_probe_uses_local_no_thinking_template_and_evaluator_token_options():
    tokenizer = Tokenizer('unused')
    request = task_probe(tokenizer, 'branch', False)
    messages, options = tokenizer.renders[0]
    assert options == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert '<function=return>' in messages[-1]['content']
    assert request['prompt'] == [10, 11, 12]
    assert request['max_tokens'] == 512
    assert request['extra_body']['return_token_ids'] is True
    assert request['extra_body']['skip_special_tokens'] is False


def test_success_uses_decoded_ids_but_keeps_special_token_text(monkeypatch, tmp_path):
    monkeypatch.setattr('scripts.preflight_graph_retry_schema.schemas', lambda: [])
    # Finish is permitted in both main probes, return is used in the branch probe.
    class CopyTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.renders.append((messages, kwargs))
            return list(messages[-1]['content'].encode())

        def decode(self, ids, skip_special_tokens):
            text = bytes(ids).decode()
            return text if skip_special_tokens else text + '<|im_end|>'

    class CopyClient(Client):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(text='server text with special markers',
                token_ids=kwargs['prompt'], finish_reason='stop')], usage=None)

    client = CopyClient()
    output = tmp_path / 'preflight.json'
    checks = run_checks(client, 'model', output, CopyTokenizer(''))
    assert len(checks) == 3 and all(c['passed'] for c in checks)
    assert checks[0]['raw_output_text'].endswith('<|im_end|>')
    assert checks[0]['decoded_response'].endswith('</function>')
    assert json.loads(output.read_text())['checks'] == checks


@pytest.mark.parametrize('text,ids,reason', [
    ('<function=search><parameter=query>unfinished', [1] * 512, 'length'),
    ('not XML', [1, 2], 'stop'),
    ('<function=search><parameter=query>x</parameter></function>', None, 'stop'),
])
def test_failed_probe_persists_evidence_and_stops_without_retry(monkeypatch, tmp_path, text, ids, reason):
    monkeypatch.setattr('scripts.preflight_graph_retry_schema.schemas', lambda: [])
    client = Client(SimpleNamespace(text=text, token_ids=ids, finish_reason=reason))
    output = tmp_path / 'preflight.json'
    with pytest.raises(ValueError):
        run_checks(client, 'model', output, Tokenizer(text))
    assert len(client.calls) == 1
    saved = json.loads(output.read_text())['checks']
    assert len(saved) == 1 and saved[0]['passed'] is False
    assert saved[0]['finish_reason'] == reason
    assert saved[0]['response'] == text
    assert saved[0]['error']
    assert saved[0]['elapsed_seconds'] >= 0


def test_backend_error_is_recorded_without_starting_remaining_probes(monkeypatch, tmp_path):
    monkeypatch.setattr('scripts.preflight_graph_retry_schema.schemas', lambda: [])
    client = Client(error=RuntimeError('backend rejected regex'))
    output = tmp_path / 'preflight.json'
    with pytest.raises(RuntimeError, match='backend rejected'):
        run_checks(client, 'model', output, Tokenizer(''))
    assert len(client.calls) == 1
    assert 'backend rejected regex' in json.loads(output.read_text())['checks'][0]['error']
