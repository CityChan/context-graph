import re
from types import SimpleNamespace

import pytest

from agents.gram_decoding import action_regex, allowed_actions
from agents.gram_memory import parse_action
from agents.structured_outputs import build_vllm_guided_decoding, build_vllm_structured_sampling_kwargs


def test_constraints_follow_retrieval_and_consumption():
    before = action_regex(allowed_actions(has_document=False, external=True, external_searches=0))
    pending = action_regex(allowed_actions(has_document=True, external=True, external_searches=1))
    consumed = action_regex(allowed_actions(has_document=False, external=True, external_searches=1))
    search = "<search>April 5 2022 core</search>"
    insert = "<memory_insert>Alice &amp; Bob</memory_insert>"
    answer = "<answer>Paris</answer>"
    assert re.fullmatch(before, search) and not re.fullmatch(before, answer)
    assert not re.fullmatch(pending, search) and re.fullmatch(pending, insert)
    assert re.fullmatch(consumed, search) and re.fullmatch(consumed, answer)
    assert not re.fullmatch(consumed, insert)


def test_progress_grammar_excludes_empty_graph_loops_and_pending_answers():
    for has_memory in (False, True):
        for count in (0, 1, 2, 98):
            for pending in (False, True):
                actions = allowed_actions(has_document=pending, external=True, external_searches=1,
                    progress_limit=2, has_memory=has_memory, memory_searches=count)
                regex = action_regex(actions)
                assert bool(re.fullmatch(regex, "<memory_search>film critic</memory_search>")) == (has_memory and count < 2)
                assert bool(re.fullmatch(regex, "<answer>Alexander</answer>")) == (not pending)
                assert bool(re.fullmatch(regex, "<search>new evidence</search>")) == (not pending)
    assert allowed_actions(has_document=False, external=True, external_searches=0,
                           progress_limit=2, has_memory=False) == ["search"]


@pytest.mark.parametrize("text", [
    "I need to refine the query.\n<search>core</search>",
    "<search>core</search> More reasoning", "<search>x</search><answer>y</answer>",
    "<search>raw & ampersand</search>", "<search><answer>x</answer></search>",
    '<search query="x">x</search>', "<search>x\x01</search>",
])
def test_regression_plain_prose_and_malformed_xml_are_not_generated(text):
    assert not re.fullmatch(action_regex(["search", "answer"]), text)
    with pytest.raises(ValueError):
        parse_action(text, external_search=True)


@pytest.mark.parametrize("text", [
    "<search>cancer May 2021</search>",
    "<think>Need more evidence.</think>\n<search>A &amp; B</search>",
    "<answer>北京 😀 &lt;title&gt;</answer>",
])
def test_legal_outputs_remain_strict_xml(text):
    assert re.fullmatch(action_regex(["search", "answer"]), text)
    assert parse_action(text, external_search=True)[1]


@pytest.mark.parametrize("modern", [False, True])
def test_regex_reaches_both_vllm_sampling_apis(modern):
    class Params:
        def __init__(self, *, regex):
            self.regex = regex
    wire = {"regex": action_regex(["search"])}
    module = SimpleNamespace(GuidedDecodingParams=Params)
    if modern:
        module.StructuredOutputsParams = Params
    result = build_vllm_structured_sampling_kwargs(wire, module)
    assert result["structured_outputs" if modern else "guided_decoding"].regex == wire["regex"]
    assert build_vllm_guided_decoding(wire, Params).regex == wire["regex"]
