#!/usr/bin/env python3
"""Fine-tune and calibrate a binary graph-state evaluator for GraphRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph_rpo import format_graph_evaluator_input


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            logits_t / log_temperature.exp().clamp_min(1e-4), labels_t
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(min=1e-4, max=100.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_frame = pd.read_parquet(args.train_file)
    validation_frame = pd.read_parquet(args.validation_file)
    required = {"question", "graph_view", "label", "task_id"}
    for name, frame in (("train", train_frame), ("validation", validation_frame)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
    split_key = "question_hash" if "question_hash" in train_frame.columns else "task_id"
    overlap = set(train_frame[split_key].astype(str)) & set(validation_frame[split_key].astype(str))
    if overlap:
        raise ValueError(f"question-disjoint split violated by {len(overlap)} question IDs")
    for name, frame in (("train", train_frame), ("validation", validation_frame)):
        if set(frame.label.astype(int)) != {0, 1}:
            raise ValueError(f"{name} split must contain both binary outcome classes")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, trust_remote_code=True
    )

    def make_dataset(frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_dict(
            {
                "text": [
                    format_graph_evaluator_input(question, view)
                    for question, view in zip(frame.question, frame.graph_view, strict=True)
                ],
                "labels": frame.label.astype(int).tolist(),
            }
        )
        return dataset.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, max_length=args.max_length
            ),
            batched=True,
            remove_columns=["text"],
        )

    train_data = make_dataset(train_frame)
    validation_data = make_dataset(validation_frame)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    prediction = trainer.predict(validation_data)
    temperature = fit_temperature(prediction.predictions, validation_frame.label.to_numpy())
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    calibration = {
        "schema_version": "contextgraph.graph_evaluator_calibration.v1",
        "temperature": temperature,
        "positive_label_id": 1,
        "validation_rows": len(validation_frame),
    }
    (args.output_dir / "graph_rpo_calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
