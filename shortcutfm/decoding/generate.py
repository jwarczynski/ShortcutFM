import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf import OmegaConf as om

from shortcutfm.config import GenerationConfig
from shortcutfm.decoding.generation_runner import (
    run_exca_job_submission,
    run_generation_with_evaluation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_generation_config(config_path: str, args_list: list[str]) -> GenerationConfig:
    """Parse and validate generation config from YAML file"""
    if not Path(config_path).exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    merged_cfg = om.merge(yaml_cfg, om.from_cli(args_list))
    merged_dict = OmegaConf.to_container(merged_cfg, resolve=True)

    # Use model_validate to create config which will trigger field validators
    cfg = GenerationConfig(**merged_dict) # type: ignore
    return cfg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m shortcutfm.decoding.generate <config_path> <cli_args>")
        sys.exit(1)

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Parse and validate generation config
    gen_cfg = parse_generation_config(yaml_path, args_list)
    logger.info("Generation Configuration:\n" + om.to_yaml(gen_cfg.model_dump()))

    if not gen_cfg.use_exca:
        # Run generation directly using the extracted module
        run_generation_with_evaluation(gen_cfg)
    else:
        # Use exca for job submission using the extracted module
        run_exca_job_submission(gen_cfg)
