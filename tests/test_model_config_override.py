import ast
from pathlib import Path
from types import SimpleNamespace


def _load_update_model_config():
    source = Path("verl/utils/model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "update_model_config"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, "verl/utils/model.py", "exec"), namespace)
    return namespace["update_model_config"]


def test_update_model_config_replaces_none_with_nested_mapping():
    update_model_config = _load_update_model_config()
    config = SimpleNamespace(rope_scaling=None)
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 32768,
    }

    update_model_config(config, {"rope_scaling": rope_scaling})

    assert config.rope_scaling == rope_scaling


def test_update_model_config_still_recurses_into_nested_config():
    update_model_config = _load_update_model_config()
    config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=40960))

    update_model_config(config, {"text_config": {"max_position_embeddings": 65536}})

    assert config.text_config.max_position_embeddings == 65536
