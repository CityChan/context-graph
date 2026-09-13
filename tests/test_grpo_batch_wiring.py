import ast
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_function(path: Path, function_name: str, namespace=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {} if namespace is None else dict(namespace)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def test_controller_minibatch_matches_worker_rollout_expansion():
    helper = _load_function(
        ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py",
        "_global_actor_mini_batch_size",
    )
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(ppo_mini_batch_size=12),
            rollout=SimpleNamespace(n=8),
        ),
        trainer=SimpleNamespace(nnodes=3),
    )

    assert helper(config) == 96
    assert 96 % helper(config) == 0


def test_actor_padding_divisor_covers_minibatch_and_three_way_dp():
    path = ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    minibatch_helper = _load_function(path, "_global_actor_mini_batch_size")
    padding_helper = _load_function(
        path,
        "_actor_padding_divisor",
        namespace={
            "math": math,
            "_global_actor_mini_batch_size": minibatch_helper,
        },
    )
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(ppo_mini_batch_size=2),
            rollout=SimpleNamespace(n=2),
        )
    )

    assert padding_helper(config, world_size=3, loss_mode="vanilla") == 12
    assert padding_helper(config, world_size=3, loss_mode="graphrpo") == 3
    assert (-13) % padding_helper(config, world_size=3, loss_mode="vanilla") == 11


def test_standard_grpo_does_not_require_project_workflow_plugin():
    helper = _load_function(
        ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py",
        "_configured_rollout_workflow",
    )
    standard_config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace())
    )
    project_config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(plugin=SimpleNamespace(workflow="search_graph"))
        )
    )

    assert helper(standard_config) is None
    assert helper(project_config) == "search_graph"


def test_workflow_is_attached_conditionally_in_train_and_validation():
    source = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert source.count("workflow = _configured_rollout_workflow(self.config)") == 2
    assert source.count("if workflow is not None:") == 2
    assert "self.config.actor_rollout_ref.rollout.plugin.workflow" not in source


def test_dummy_padding_is_loss_inert_and_not_node_count_based():
    source = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "padding_divisor = _actor_padding_divisor(" in source
    assert "return math.lcm(_global_actor_mini_batch_size(config), world_size)" in source
    assert "self.actor_rollout_wg.world_size" in source
    assert "ppo_mini_batch_size * self.config.trainer.nnodes" not in source
    assert 'dummy_sample.batch["response_mask"] = torch.zeros_like' in source
    assert 'if "overlong_mask" in dummy_sample.batch:' in source


def test_dummy_padding_is_excluded_from_user_facing_metrics():
    helper = _load_function(
        ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py",
        "_without_dummy_trajectories",
    )

    class FakeBatch:
        meta_info = {"gen_uid_dummy": "dummy"}
        non_tensor_batch = {"gen_uid": ["real-a", "dummy", "real-b"]}

        def __getitem__(self, indices):
            return indices

    assert helper(FakeBatch()) == [0, 2]


def test_standard_grpo_does_not_require_contextgraph_rollout_mask():
    source = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert 'if "mask_rollout" in batch.batch:' in source
    assert 'mask_rollout = batch.batch["mask_rollout"]' in source
    assert "metrics['optimization_masked_rollouts'] = optimization_masked_rollouts" in source


def test_actor_preserves_overlong_mask_for_policy_loss():
    source = (ROOT / "verl" / "workers" / "actor" / "dp_actor.py").read_text(encoding="utf-8")
    update_policy = source[source.index("    def update_policy(") :]
    append = 'select_keys.append("overlong_mask")'

    assert 'if "overlong_mask" in data.batch.keys():' in update_policy
    assert update_policy.index(append) < update_policy.index("data = data.select(")


def test_search7b_launcher_uses_rollout_aligned_batch_and_skill_bank():
    launcher = (
        ROOT / "scripts" / "train_nq_hotpot_grpo_search7b_sft_4node_100step.sh"
    ).read_text(encoding="utf-8")
    baseline = (
        ROOT / "scripts" / "train_bc_baseline_8b_4node_24h_v3_32k.sh"
    ).read_text(encoding="utf-8")

    assert "export TRAIN_BATCH_SIZE=48" in launcher
    assert "export PPO_MINI_BATCH_SIZE=48" in launcher
    assert "export ROLLOUT_N=4" in launcher
    assert "export MAX_TURN=4" in launcher
    assert "export LR_WARMUP_STEPS_RATIO=0.1" in launcher
    assert "export ACTOR_KL_LOSS_COEF=0.001" in launcher
    assert "export USE_SKILLS_ONLY_MEMORY=True" in launcher
    assert "claude_style_skills_search.json" in launcher
    assert 'optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO"' in baseline
    assert 'plugin.use_skills_only_memory="$USE_SKILLS_ONLY_MEMORY"' in baseline
