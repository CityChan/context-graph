import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.prompts import create_chat
from envs.math_env import MathEnv, extract_numeric_answer, has_contextgraph_finish_format


def _item(answer="42"):
    return SimpleNamespace(
        non_tensor_batch={
            "extra_info": {
                "query": "What is 6 times 7?",
                "answer": answer,
                "workflow": "math_graph",
            }
        }
    )


def test_math_env_is_selected_for_math_ability():
    source = Path("agents/utils.py").read_text(encoding="utf-8")
    assert "from envs.math_env import MathEnv" in source
    assert "ability.lower() == 'math'" in source
    assert "'GSM8K' in ability" in source


def test_numeric_answer_normalizes_commas_and_decimal_equivalence():
    assert extract_numeric_answer("#### 1,234.0") == extract_numeric_answer("1234")


def test_numeric_answer_prefers_explicit_submission_over_trailing_numbers():
    cases = [
        ("answer>13,140 explanation>computed correctly confidence>96% - 97%/90%", "13140"),
        ("<answer>220</answer> explanation>66 + 132 + 22 = 220 confidence>99", "220"),
        ('<finish> answer="75" explanation="25 + 50 = 75" confidence="9" </finish>', "75"),
        ("YOUR BEST ANSWER: 15 EXPLANATION: 90 - 75 = 15 CONFIDENCE: 100%", "15"),
        ("answer=32850 explanation=the calculation later mentions 13140", "32850"),
        ("Answer: Mike will have $800 after buying the shirt. Confidence: 100%", "800"),
        ("The final answer is 99 bananas per monkey. Confidence: 100%", "99"),
        ("Each monkey receives 1188 / 12 = 99 bananas. My confidence is 100%.", "99"),
    ]

    for text, expected in cases:
        assert extract_numeric_answer(text) == Decimal(expected)


def test_numeric_answer_falls_back_to_last_number_without_answer_marker():
    assert extract_numeric_answer("First compute 6 times 7, giving 42") == Decimal("42")


def test_math_env_finish_and_exact_reward():
    env = MathEnv(SimpleNamespace(), None, "math")
    asyncio.run(env.init_env(_item()))
    result = asyncio.run(env.run_action(
        "<function=finish><parameter=answer>#### 42.0</parameter></function>"
    ))
    reward = asyncio.run(env.get_reward(None, [], None))

    assert result == {"action": "finish"}
    assert reward[1] == 1
    assert env.judge_audit[-1]["judge_method"] == "gsm8k_exact"


def test_math_env_rejects_wrong_answer():
    env = MathEnv(SimpleNamespace(), None, "math")
    asyncio.run(env.init_env(_item()))
    asyncio.run(env.run_action(
        "<function=finish><parameter=answer>41</parameter></function>"
    ))
    assert asyncio.run(env.get_reward(None, [], None))[1] == 0


def test_math_env_qerl_aligned_reward_uses_contextgraph_finish_format():
    config = SimpleNamespace(
        plugin={
            "math_correctness_reward_weight": 2.0,
            "math_format_reward_weight": 0.2,
        }
    )
    env = MathEnv(config, None, "math")
    asyncio.run(env.init_env(_item()))
    response = "<function=finish><parameter=answer>42</parameter></function>"
    asyncio.run(env.run_action(response))
    reward = asyncio.run(env.get_reward(None, [], None))

    assert has_contextgraph_finish_format(response)
    assert reward[1] == pytest.approx(2.2)
    assert env.stats["math_correctness_reward"] == pytest.approx(2.0)
    assert env.stats["math_format_reward"] == pytest.approx(0.2)
    assert env.judge_audit[-1]["total_reward"] == pytest.approx(2.2)


def test_math_env_format_reward_does_not_require_correctness():
    config = SimpleNamespace(
        plugin={
            "math_correctness_reward_weight": 2.0,
            "math_format_reward_weight": 0.2,
        }
    )
    env = MathEnv(config, None, "math")
    asyncio.run(env.init_env(_item()))
    asyncio.run(env.run_action(
        "<function=finish><parameter=answer>41</parameter></function>"
    ))

    assert asyncio.run(env.get_reward(None, [], None))[1] == pytest.approx(0.2)


def test_math_env_synthetic_finish_can_be_correct_without_format_reward():
    config = SimpleNamespace(
        plugin={
            "math_correctness_reward_weight": 2.0,
            "math_format_reward_weight": 0.2,
        }
    )
    env = MathEnv(config, None, "math")
    asyncio.run(env.init_env(_item(answer="800")))
    env.emergency_finish_wrapped = True
    asyncio.run(env.run_action(
        "<function=finish><parameter=answer>Answer: Mike will have $800. Confidence: 100%</parameter></function>"
    ))

    assert asyncio.run(env.get_reward(None, [], None))[1] == pytest.approx(2.0)
    assert env.stats["math_correctness"] == 1
    assert env.stats["math_format_valid"] == 0


def test_math_graph_prompt_exposes_contextgraph_protocol():
    exposed = create_chat("What is 6 times 7?", "math_graph", expose_graph_tools=True)
    controlled = create_chat("What is 6 times 7?", "math_graph", expose_graph_tools=False)
    exposed_text = "\n".join(message["content"] for message in exposed)
    controlled_text = "\n".join(message["content"] for message in controlled)

    assert "FUNCTION #" in exposed_text
    assert "finish" in exposed_text and "branch" in exposed_text and "merge" in exposed_text
    assert ": search ----" not in exposed_text and ": open_page ----" not in exposed_text
    assert "[GRAPH ACTION MODE]" in controlled_text
    assert "merge" not in controlled_text
