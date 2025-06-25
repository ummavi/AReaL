"""
Sync Tool-Use Search Agent Training Script

This is a minimal adaptation of main_sync_ppo.py for our tool-use agent.
"""

import dataclasses
import datetime
import os
from typing import Dict

import hydra
import yaml
from omegaconf import MISSING, OmegaConf

from realhf.api.quickstart.entrypoint import kind_reminder
# from realhf.base.constants import init_constants  # Not needed

# We'll create a sync version of our config
from realhf.experiments.common.ppo_math_exp import PPOMATHConfig
from realhf.api.core.config import AgentAbstraction, EnvServiceAbstraction
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.experiments.common.utils import asdict
from typing import Any

@dataclasses.dataclass
class SyncToolUseConfig(PPOMATHConfig):
    """Sync version of our tool-use config."""

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

from training.utils import run_experiment


@hydra.main(version_base=None, config_path="configs", config_name="sync-ppo")
def main_sync_tool_use_training(args):
    # NOTE: we import logging here to avoid hydra logging overwrite
    import realhf.base.logging as logging

    logger = logging.getLogger("quickstart", "colored")

    # Overwrite the python dataclass configuration with yaml
    default_args = OmegaConf.structured(SyncToolUseConfig)
    args = OmegaConf.merge(default_args, args)
    args: SyncToolUseConfig = OmegaConf.to_object(args)

    # Set experiment trial name.
    exp_name = args.experiment_name
    if args.trial_name == MISSING:
        args.trial_name = trial_name = (
            f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
    else:
        trial_name = args.trial_name

    if args.mode != "ray":
        raise RuntimeError("This script only supports the `ray` mode.")

    # init_constants(args)  # Not needed

    # from realhf.base.constants import LOG_ROOT
    LOG_ROOT = "/tmp/areal_logs"  # Default log root

    # Save overwritten configuration to yaml
    config_save_path = os.path.join(
        LOG_ROOT, args.experiment_name, args.trial_name, "config.yaml"
    )
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, "w") as f:
        config_dict: Dict = dataclasses.asdict(args)
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
        )

    logger.info(f"Running sync tool-use-search experiment.")
    kind_reminder("sync-tool-use", logger, args)
    run_experiment(args, exp_name, trial_name)


if __name__ == "__main__":
    main_sync_tool_use_training()