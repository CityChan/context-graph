from pathlib import Path

from scripts.gsm8k_qerl_xml_reward import compute_score
from scripts.prepare_gsm8k_grpo_data import convert_example


LAUNCHER = Path("scripts/train_gsm8k_verl_grpo_qerlmatched_lora32_200step.sh")
AGENT_LOOP = Path("verl/experimental/agent_loop/agent_loop.py")


def test_qerl_xml_reward_matches_upstream_string_semantics():
    correct = compute_score("openai/gsm8k", "<think>x</think><answer>2</answer>", "2")
    unit_false_negative = compute_score(
        "openai/gsm8k", "<think>x</think><answer>2 gallons</answer>", "2"
    )
    malformed = compute_score("openai/gsm8k", "2", "2")

    assert correct == {
        "score": 2.2,
        "correctness": 1.0,
        "correctness_reward": 2.0,
        "soft_format_valid": 1.0,
        "soft_format_reward": 0.2,
    }
    assert unit_false_negative["score"] == 0.2
    # QeRL's split-based extractor treats the entire response as the answer
    # when XML tags are absent, so correctness can still score without format.
    assert malformed["score"] == 2.0
    assert malformed["soft_format_reward"] == 0.0


def test_qerl_xml_prompt_matches_expected_two_message_shape():
    row = convert_example(
        {"question": "What is 1+1?", "answer": "work\n#### 2"},
        "train",
        0,
        prompt_style="qerl_xml",
    )

    assert [message["role"] for message in row["prompt"]] == ["system", "user"]
    assert "<think>" in row["prompt"][0]["content"]
    assert row["prompt"][1]["content"] == "What is 1+1?"
    assert row["reward_model"]["ground_truth"] == "2"


def test_single_turn_launcher_excludes_contextgraph_agent_path():
    source = LAUNCHER.read_text()

    required = [
        "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}",
        "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}",
        "ROLLOUT_N=${ROLLOUT_N:-16}",
        "LORA_RANK=${LORA_RANK:-32}",
        "OPTIMIZER=${OPTIMIZER:-AdamW8bit}",
        "DATA_PROMPT_STYLE=${DATA_PROMPT_STYLE:-qerl_xml}",
        "gsm8k_qerl_xml_reward.py",
        "smoke_train_gsm8k_grpo_1node_10step.sh",
    ]
    for setting in required:
        assert setting in source

    assert "context_graph_isolated_agent" not in source
    assert "graph_rpo_credit_backend" not in source


def test_async_agent_loop_preserves_grpo_grouping_ids():
    source = AGENT_LOOP.read_text()

    assert 'for identity_key in ("uid", "gen_uid"):' in source
    assert 'output.extra_fields.setdefault(identity_key, kwargs[identity_key])' in source
