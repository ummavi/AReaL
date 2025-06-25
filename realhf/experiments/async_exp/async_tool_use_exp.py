"""
Tool-Use Search Agent Experiment Configuration for AReAL

This is a minimal wrapper around AsyncRLExperimentConfig that overrides
the hardcoded math agent with our tool-use search agent.

This preserves our Agent-as-a-Service architecture while working around
AReAL's hardcoded experiment configurations.
"""

import dataclasses
from typing import Any

from realhf.api.core.config import AgentAbstraction, EnvServiceAbstraction
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.experiments.async_exp.async_rl_exp import AsyncRLExperimentConfig
from realhf.experiments.common.ppo_math_exp import PPOMATHConfig
from realhf.experiments.common.utils import asdict


@dataclasses.dataclass
class AsyncToolUseConfig(AsyncRLExperimentConfig, PPOMATHConfig):
    """
    Tool-Use Search experiment config that uses our registered agent
    instead of the hardcoded math agent.
    """

    @property
    def agent(self) -> AgentAbstraction:
        """Use our registered tool-use search agent instead of math agent."""
        return AgentAbstraction(
            "tool-use-search",  # Our registered agent name
            args=dict(
                gconfig=self.generation_config,
                tokenizer_path=self.actor.path,
                success_rate_lb=self.success_rate_lb,
                success_rate_ub=self.success_rate_ub,
                reward_scaling=self.ppo.reward_output_scaling,
                reward_bias=self.ppo.reward_output_bias,
            ),
        )

    @property
    def env(self) -> EnvServiceAbstraction:
        """Use null environment for now (our agent handles environment internally)."""
        return EnvServiceAbstraction(
            "null",  # Use null environment since our agent is self-contained
            args=dict(dataset_path=self.dataset.path)
        )

    @property
    def generation_config(self) -> GenerationHyperparameters:
        """Generation configuration for our tool-use agent."""
        return GenerationHyperparameters(**asdict(self.ppo.gen)).new(n=self.group_size)

    @property
    def gen_backend_args(self) -> Any:
        return self.actor.sglang