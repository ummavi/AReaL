"""
Tool-Use Search Agent Training Script

This is a minimal adaptation of main_async_ppo.py that uses our
AsyncToolUseConfig instead of the hardcoded AsyncPPOMATHConfig.

This preserves our Agent-as-a-Service architecture while working around
AReAL's hardcoded experiment configurations.
"""

import dataclasses
import datetime
import os
from typing import Dict

import hydra
import yaml
from omegaconf import MISSING, OmegaConf

from realhf.api.quickstart.entrypoint import kind_reminder
from realhf.base.constants import init_constants
from realhf.experiments.async_exp.async_tool_use_exp import AsyncToolUseConfig
from training.utils import run_experiment


@hydra.main(version_base=None, config_path="configs", config_name="async-ppo")
def main_tool_use_training(args):
    # NOTE: we import logging here to avoid hydra logging overwrite
    import realhf.base.logging as logging

    logger = logging.getLogger("quickstart", "colored")

    # Overwrite the python dataclass configuration with yaml
    default_args = OmegaConf.structured(AsyncToolUseConfig)
    args = OmegaConf.merge(default_args, args)
    args: AsyncToolUseConfig = OmegaConf.to_object(args)

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

    init_constants(args)

    from realhf.base.constants import LOG_ROOT

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

    logger.info(f"Running tool-use-search experiment.")
    kind_reminder("async-tool-use", logger, args)
    run_experiment(args, exp_name, trial_name)


if __name__ == "__main__":
    main_tool_use_training()