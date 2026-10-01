import pytest

pytest.importorskip("ray")

from verl.utils.logging_utils import redact_ray_init_kwargs as _redact_ray_init_kwargs


def test_redact_ray_init_kwargs_hides_sensitive_env_values():
    kwargs = {
        "num_cpus": None,
        "runtime_env": {
            "env_vars": {
                "OPENAI_API_KEY": "openai-secret",
                "WANDB_API_KEY": "wandb-secret",
                "ACCESS_TOKEN": "token-secret",
                "LOCAL_SEARCH_URL": "http://127.0.0.1:18999",
            }
        },
    }

    redacted = _redact_ray_init_kwargs(kwargs)

    assert redacted["runtime_env"]["env_vars"] == {
        "OPENAI_API_KEY": "[REDACTED]",
        "WANDB_API_KEY": "[REDACTED]",
        "ACCESS_TOKEN": "[REDACTED]",
        "LOCAL_SEARCH_URL": "http://127.0.0.1:18999",
    }
    assert kwargs["runtime_env"]["env_vars"]["OPENAI_API_KEY"] == "openai-secret"
