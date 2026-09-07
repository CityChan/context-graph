import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from agents.prompts import create_chat
from envs.math_env import MathEnv, extract_numeric_answer


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
