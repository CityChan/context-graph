from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_contextgraph_sft_data.py"


def test_preflight_accepts_an_explicit_loss_mask_mode():
    text = SCRIPT.read_text()
    assert '"--loss-mask-mode"' in text
    assert '"loss_mask_mode": args.loss_mask_mode' in text
    assert '"loss_mask_mode": args.loss_mask_mode' in text
