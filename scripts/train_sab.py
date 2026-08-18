"""Evaluation/training entry point for code-domain benchmarks.

The shared ReAct, FoldAgent, and ContextGraph code loops are registered in
``verl.experimental.agent_loop.code_agent_loop`` so this executable module can
be imported safely by ScienceAgentBench and DiscoveryBench entry points.
"""

from verl.experimental.agent_loop.code_agent_loop import (
    ContextGraphCodeIsolatedLoop,
    FoldAgentCodeLoop,
    ReactAgentCodeLoop,
)

# Keep explicit references for discoverability and static packaging tools.
_ = [ReactAgentCodeLoop, FoldAgentCodeLoop, ContextGraphCodeIsolatedLoop]


def main():
    """Run VERL PPO with all three code-domain agent loops registered."""
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
