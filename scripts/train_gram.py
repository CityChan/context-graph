"""VERL GRPO for document-stream and BC-P GRAM; memory helpers stay frozen."""
import json
from uuid import uuid4

from agents.gram_agent import GramConfig, MemoryBackend, make_policy, run_episode, score_episode


def validate_training_config(config):
    algorithm, rollout = config.algorithm, config.actor_rollout_ref.rollout
    if algorithm.adv_estimator != "foldgrpo" or algorithm.foldgrpo_process_reward_mode != "relative_extrema":
        raise ValueError("GRAM requires episode-deduplicated GRPO (foldgrpo / relative_extrema, Q=0)")
    if algorithm.fix_bad_positive_adv or algorithm.use_kl_in_reward:
        raise ValueError("GRAM uses terminal task+format reward and actor KL, without reward-side KL/masking")
    gram = rollout.plugin.gram
    if not gram.memory_endpoint or not gram.memory_model or not gram.memory_revision:
        raise ValueError("Configure an explicit frozen memory model and revision")
    if not gram.frozen_memory_acknowledged:
        raise ValueError("Memory endpoint must be a separately served frozen checkpoint")


async def process_item(fields, context, memory=None, audit=None):
    import numpy as np
    from agents.utils import AgentLoopMetrics, AgentLoopOutput
    from omegaconf import OmegaConf
    def scalar(value):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError("Expected one rollout item")
            return value.reshape(-1)[0]
        return value
    validate_training_config(context.config)
    rollout = context.config.actor_rollout_ref.rollout
    settings = rollout.plugin.gram
    benchmark = settings.get("benchmark", "document-stream")
    if benchmark not in {"document-stream", "bcp"}:
        raise ValueError(f"Unsupported GRAM training benchmark: {benchmark}")
    task = json.loads(scalar(fields["extra_info"])["gram_task_json"])
    # Ground truth stays outside the executor and every helper call.
    answers = json.loads(scalar(fields["reward_model"])["ground_truth"])
    uid = scalar(fields.get("uid", task["task_id"]))
    generation_id = scalar(fields.get("gen_uid")) or uuid4().hex
    config = GramConfig(**OmegaConf.to_container(settings.episode, resolve=True))
    owned = memory is None
    if owned:
        memory = MemoryBackend(settings.memory_endpoint, settings.memory_model, audit=audit,
                               embedding_endpoint=settings.get("embedding_endpoint"),
                               embedding_model=settings.get("embedding_model"))
    retrieval = None
    try:
        if benchmark == "bcp":
            from agents.gram_bcp import BcpRetrieval, score_bcp
            if len(answers) != 1:
                raise ValueError("BC-P requires one private reference answer")
            retrieval = BcpRetrieval()
        result = await run_episode(task, make_policy(context.llm_client, context.tokenizer, rollout),
                                   memory, config, audit, retrieval=retrieval)
        if benchmark == "bcp":
            graded = await score_bcp(task["question"], answers[0], result["prediction"])
            outcome = graded["score"]
            if outcome not in (0, 1):
                raise ValueError("BC-P task reward must be binary")
            scores = {"task_reward": outcome, "format_reward": result["format_reward"],
                      "training_reward": outcome + config.process_weight * result["format_reward"]}
            if audit:
                audit({"kind": "bcp_reward", "task_id": task["task_id"],
                       "validation": not getattr(context, "is_train", True),
                       "external_searches": result["external_searches"], **scores,
                       "judge_audit": graded["judge_audit"]})
        else:
            scores = score_episode(result, answers, config.process_weight)
            scores["task_reward"] = scores["answer_f1"]
    finally:
        import asyncio
        await asyncio.gather(*([memory.aclose()] if owned else []),
                             *([retrieval.aclose()] if retrieval is not None else []))
    outputs = []
    for segment in result.pop("segments"):
        outputs.append(AgentLoopOutput(
            prompt_ids=segment["prompt_ids"], response_ids=segment["response_ids"],
            response_mask=segment["response_mask"], response_logprobs=segment["response_logprobs"],
            multi_modal_data={}, metrics=AgentLoopMetrics(), num_turns=segment["num_turns"],
            reward_score=scores["training_reward"], extra_fields={
                "uid": uid, "gen_uid": generation_id, "mask_rollout": False,
                "is_finish": result["termination_reason"] == "answer",
                "termination_reason": result["termination_reason"], "messages": segment["messages"],
                "process_reward_mask": [0.0]*len(segment["response_ids"]),
                "env_stats": {**scores, "main_turn": result["steps"],
                              "gram_documents_consumed": result["documents_consumed"],
                              "gram_policy_tokens": result["policy_tokens"]}}))
    return outputs


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--check-data":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: --check-data TRAIN.parquet VALIDATION.parquet")
        check_data(sys.argv[2], sys.argv[3])
        print("GRAM_TRAIN_DATA_OK")
        return
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


def check_data(train, validation):
    from pathlib import Path
    import pandas as pd
    from agents.gram_memory import entity_key
    from scripts.prepare_gram_data import sha256
    seen_ids, seen_questions = set(), set()
    for file, split in ((train, "train"), (validation, "validation")):
        path = Path(file)
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest["split"] != split or sha256(path) != manifest["files"][path.name]:
            raise ValueError("Training data split/hash mismatch")
        frame = pd.read_parquet(path)
        if not len(frame):
            raise ValueError("Empty training/validation file")
        tasks = [json.loads(extra["gram_task_json"]) for extra in frame.extra_info]
        ids = {task["task_id"] for task in tasks}
        questions = {entity_key(task["question"]) for task in tasks}
        if len(ids) != len(tasks) or ids & seen_ids or questions & seen_questions:
            raise ValueError("Duplicate tasks or train/validation overlap")
        seen_ids.update(ids)
        seen_questions.update(questions)


if __name__ == "__main__":
    main()
