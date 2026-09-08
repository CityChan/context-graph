# TRL agent backend

This backend runs the existing FoldAgent and isolated ContextGraph loops on top
of QeRL's TRL trainer and colocated vLLM engine. It is intended for single-GPU
throughput experiments and framework comparisons.

The adapter preserves the agent-training semantics that a flattened completion
would lose:

- all assistant generations, tool observations, and controller prompts remain
  in the model attention context;
- only policy-generated tokens are included in the policy loss;
- main and branch streams keep their shared `gen_uid` episode identity;
- FoldGRPO consumes token process labels;
- GraphRPO deduplicates terminal rewards by episode and applies exact
  `1 / (|Q| K_q |M_g|)` token weights;
- variable branch counts are split without dropping remainder trajectories;
- concurrent agent turns are coalesced into serialized batched vLLM calls.

## Vista setup

Run the environment setup once:

```bash
cd /work/09281/chc_1996/vista/context-graph && bash scripts/setup_graphtrl_env.sh
```

Then run a 10-step smoke inside a one-GPU GH200 allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && bash scripts/train_gsm8k_trl_foldagent_lora32_10step.sh
```

or:

```bash
cd /work/09281/chc_1996/vista/context-graph && bash scripts/train_gsm8k_trl_contextgraph_lora32_10step.sh
```

Set `TOTAL_TRAINING_STEPS=200` to extend either run. Both launchers use the same
Qwen2.5-1.5B, LoRA 32/32, AdamW8bit, G16, and colocated-vLLM training settings as
the existing QeRL control.

The initial GraphRPO backend supports graph credits computed during rollout:
`old_policy_counterfactual_qa` and `external_evaluator`. The two answer-
likelihood backends require a trainer-side reference-answer scoring pass and
fail explicitly rather than silently running without edit credit.
