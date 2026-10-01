import json
from pathlib import Path

import pytest

from scripts.evaluation_records import read_evaluation
from scripts.export_audited_results import audit_arm_pilot, audit_run, checkpoint_revision, digest, export


def fixture(root):
    root.mkdir(exist_ok=True)
    for rank in range(3):
        manifest = dict(source_sha256="datahash", indices=[0, 1, 2], rank=rank,
                        commit="commit", model="test-model", model_path="/private/snapshots/revision",
                        method="react", seed=42, judge_model="judge",
                        config={"actor_rollout_ref": {"rollout": {"prompt_length": 10, "response_length": 20,
                                                                  "plugin": {"api_key": "DO-NOT-PUBLISH", "max_turn": 5}}}})
        (root / f"manifest-{rank}.json").write_text(json.dumps(manifest), encoding="utf-8")
        row = dict(source_index=rank, task_reward=int(rank == 0), is_finish=True, status="ok",
                   private_prompt="DO-NOT-PUBLISH", env_stats={"judge_parse_failure": 0})
        (root / f"results-{rank}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    _, _, summary = read_evaluation(root)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def test_export_reconciles_score_and_omits_sensitive_content(tmp_path):
    root = fixture(tmp_path / "run")
    run = audit_run(root)
    assert run["summary"]["task_accuracy"] == pytest.approx(1 / 3)
    assert run["checkpoint_revision"] == "revision"
    assert "DO-NOT-PUBLISH" not in json.dumps(run)
    assert "/private/" not in json.dumps(run)
    assert len(run["source_files"]) == 7
    assert checkpoint_revision("/private/custom-model") is None


@pytest.mark.parametrize("mutation,expected", [("duplicate", "duplicated"), ("provenance", "provenance"),
                                             ("summary", "disagrees"), ("empty", "Empty"),
                                             ("nonbinary", "Nonbinary")])
def test_export_rejects_inconsistent_evidence(tmp_path, mutation, expected):
    root = fixture(tmp_path / "run")
    if mutation == "duplicate":
        path = root / "results-0.jsonl"
        path.write_text(path.read_text() * 2)
    elif mutation == "summary":
        path = root / "summary.json"
        value = json.loads(path.read_text())
        value["task_successes"] = 3
        path.write_text(json.dumps(value))
    elif mutation == "nonbinary":
        path = root / "results-0.jsonl"
        value = json.loads(path.read_text())
        value["task_reward"] = 0.5
        path.write_text(json.dumps(value))
    else:
        path = root / "manifest-0.json"
        value = json.loads(path.read_text())
        value["seed" if mutation == "provenance" else "indices"] = 9 if mutation == "provenance" else []
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=expected):
        audit_run(root)


def test_missing_evidence_is_listed_as_exclusion(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    root = fixture(source / "run")
    (root / "results-1.jsonl").unlink()
    payload = export(source, tmp_path / "published")
    assert not payload["runs"]
    assert payload["excluded"][0]["run_id"] == "run"


def test_arm_pilot_checks_patch_and_prediction_hash_without_claiming_full_score(tmp_path):
    instance = "test__repo-1"
    (tmp_path / "grading-arm").mkdir()
    patch_dir = tmp_path / "instances" / instance
    patch_dir.mkdir(parents=True)
    (patch_dir / "model.patch").write_bytes(b"")
    predictions = [{"instance_id": instance, "model_patch": ""}]
    (tmp_path / "predictions.jsonl").write_text(json.dumps(predictions[0]) + "\n")
    sha = digest(tmp_path / "predictions.jsonl")
    manifest = dict(instance_ids=[instance], dataset={"count": 500}, harness_commit="harness",
                    method="foldagent", model="model", model_path="/private/snapshots/revision",
                    seed=42, commit="code", config={"actor_rollout_ref": {"rollout": {"prompt_length": 8192, "response_length": 24576}}})
    grading = dict(instance_id=instance, runtime="apptainer-arm-pilot", official_x86_result=False,
                   count=1, grading_complete=True, dataset=manifest["dataset"], harness_commit="harness",
                   predictions_sha256=sha, sif_sha256="image", empty_patch=True, resolved=0)
    generation = dict(method="foldagent", predictions_sha256=sha, selected=1, generated=1,
                      generation_errors=0, empty_patches=1)
    row = dict(instance_id=instance, status="generated", image={"sif_sha256": "image"},
               patch_bytes=0, seed=123, env_stats={})
    for name, value in [("manifest.json", manifest), ("grading-arm/summary.json", grading),
                        ("generation_summary.json", generation), ("results.jsonl", row)]:
        (tmp_path / name).write_text(json.dumps(value))
    pilot = audit_arm_pilot(tmp_path)
    assert pilot["grading"]["count"] == 1
    assert pilot["grading"]["dataset"]["count"] == 500
    assert pilot["grading"]["official_x86_result"] is False
    (patch_dir / "model.patch").write_bytes(b"unexpected change")
    with pytest.raises(ValueError, match="inconsistent"):
        audit_arm_pilot(tmp_path)
