from scripts.prepare_gsm8k_grpo_data import convert_example, extract_ground_truth


def test_extract_ground_truth_normalizes_commas():
    assert extract_ground_truth("work\n#### 1,234") == "1234"


def test_convert_example_uses_verl_gsm8k_schema():
    row = convert_example(
        {"question": "What is 6 times 7?", "answer": "6 * 7 = 42\n#### 42"},
        "train",
        3,
    )

    assert row["data_source"] == "openai/gsm8k"
    assert row["prompt"][0]["role"] == "user"
    assert 'after "####"' in row["prompt"][0]["content"]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "42"}
    assert row["extra_info"] == {"split": "train", "index": 3}
