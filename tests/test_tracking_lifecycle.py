import importlib.util
import sys
from pathlib import Path


def _load_tracking_module():
    path = Path(__file__).resolve().parents[1] / "verl/utils/tracking.py"
    spec = importlib.util.spec_from_file_location("tracking_lifecycle_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tracking_finish_is_idempotent():
    tracking_module = _load_tracking_module()

    class FakeWandb:
        def __init__(self):
            self.exit_codes = []

        def finish(self, exit_code):
            self.exit_codes.append(exit_code)

    fake_wandb = FakeWandb()
    tracking = tracking_module.Tracking.__new__(tracking_module.Tracking)
    tracking._finished = False
    tracking.logger = {"wandb": fake_wandb}

    tracking.finish(exit_code=0)
    tracking.finish(exit_code=1)

    assert fake_wandb.exit_codes == [0]


def test_ppo_trainer_explicitly_finishes_tracking_before_return():
    source = (
        Path(__file__).resolve().parents[1] / "verl/trainer/ppo/ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert source.count("logger.finish()") >= 2
