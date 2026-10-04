import asyncio
import json
from types import SimpleNamespace

import pytest


def test_prepare_bcp_rl_private_labels_and_full_source_overlap(tmp_path):
    import pandas as pd
    from scripts.prepare_gram_bcp_rl import prepare_bcp
    train, val = tmp_path / "train.parquet", tmp_path / "test.parquet"
    pd.DataFrame([{"extra_info": {"query": f"train {i}", "answer": f"PRIVATE {i}"}} for i in range(6)]).to_parquet(train)
    pd.DataFrame([{"extra_info": {"query": f"test {i}", "answer": "PRIVATE test"}} for i in range(3)]).to_parquet(val)
    prepare_bcp(tmp_path / "prepared", train, val)
    frame = pd.read_parquet(tmp_path / "prepared/train/data.parquet")
    assert len(frame) == 4
    for row in frame.to_dict("records"):
        assert "PRIVATE" not in str(row["prompt"]) + str(row["extra_info"])
        assert json.loads(row["extra_info"]["gram_task_json"])["documents"] == []
        assert "PRIVATE" in row["reward_model"]["ground_truth"]
    manifest = json.loads((tmp_path / "prepared/validation/manifest.json").read_text())
    assert manifest["count"] == 2 and manifest["source_role"] == "bc_test"
    pd.DataFrame([{"extra_info": {"query": "train 5", "answer": "x"}},
                  {"extra_info": {"query": "new q", "answer": "x"}}]).to_parquet(val)
    with pytest.raises(ValueError, match="overlap"):
        prepare_bcp(tmp_path / "bad", train, val)


@pytest.mark.parametrize("failure", [None, "retrieval", "judge"])
def test_bcp_rl_uses_retrieval_judge_private_labels_and_closes_on_failure(monkeypatch, failure):
    from omegaconf import OmegaConf
    from tests.test_session_restart import Tokenizer, Client
    from tests.test_gram import Helper
    from scripts.train_gram import process_item
    import agents.gram_bcp as bcp
    opened = []
    class Retrieval:
        closed = False
        def __init__(self):
            opened.append(self)
        async def __call__(self, operation, content):
            assert operation == "search"
            if failure == "retrieval":
                raise RuntimeError("retriever unavailable")
            return "Book by Alice, born in Paris"
        async def aclose(self):
            self.closed = True
    async def judge(question, answer, prediction):
        assert answer == "PRIVATE_REFERENCE" and prediction == "Paris"
        if failure == "judge":
            raise RuntimeError("judge unavailable")
        return {"score": 1, "judge_audit": [{"judge_method": "llm_judge"}]}
    monkeypatch.setattr(bcp, "BcpRetrieval", Retrieval)
    monkeypatch.setattr(bcp, "score_bcp", judge)
    config = OmegaConf.create({"algorithm": {"adv_estimator": "foldgrpo", "foldgrpo_process_reward_mode": "relative_extrema",
                                           "fix_bad_positive_adv": False, "use_kl_in_reward": False},
        "actor_rollout_ref": {"rollout": {"prompt_length": 12000, "response_length": 1024, "plugin": {
            "enable_summary": False, "gram": {"benchmark": "bcp", "memory_endpoint": "http://frozen", "memory_model": "helper",
            "memory_revision": "fixed", "frozen_memory_acknowledged": True, "episode": {"max_steps": 4, "max_step_tokens": 512, "action_decoding": "xml_regex"}}}}}})
    client = Client(["<search>Book author</search>", "<memory_insert>Book by Alice</memory_insert>", "<answer>Paris</answer>"])
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), llm_client=client, is_train=True)
    fields = {"extra_info": {"gram_task_json": json.dumps({"task_id": "train-0", "question": "Where?", "documents": []})},
              "uid": "q", "gen_uid": "episode", "reward_model": {"ground_truth": json.dumps(["PRIVATE_REFERENCE"])}}
    audit = []
    if failure:
        with pytest.raises(RuntimeError, match="unavailable"):
            asyncio.run(process_item(fields, context, memory=Helper(), audit=audit.append))
    else:
        outputs = asyncio.run(process_item(fields, context, memory=Helper(), audit=audit.append))
        assert len(outputs) == 3
        assert all(out.reward_score == pytest.approx(1.1) for out in outputs)
        assert all(out.extra_fields["env_stats"]["task_reward"] == 1 for out in outputs)
        assert audit[-1]["kind"] == "bcp_reward" and audit[-1]["external_searches"] == 1
        assert not audit[-1]["validation"]
        for out, (prefix, kwargs) in zip(outputs, client.calls):
            import re
            sampled = [token for token, mask in zip(out.response_ids, out.response_mask) if mask]
            assert sampled[-1] == context.tokenizer.eos_token_id
            text = context.tokenizer.decode(sampled[:-1])
            assert re.fullmatch(kwargs["structured_outputs"]["regex"], text)
            assert all(logp == -.125 for logp, mask in zip(out.response_logprobs, out.response_mask) if mask)
            assert (out.prompt_ids + out.response_ids)[:len(prefix)] == prefix
            assert "PRIVATE_REFERENCE" not in str(kwargs["messages"])
            assert not any(out.extra_fields["process_reward_mask"])
    assert opened[0].closed
