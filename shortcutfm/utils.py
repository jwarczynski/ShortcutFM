import sys

from omegaconf import DictConfig, OmegaConf as om


def parse_args(defult_config_path="configs/default.yaml"):
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Load default configuration (hardcoded defaults)
    with open(defult_config_path, "r") as f:
        default_cfg = om.load(f)

    with open(yaml_path, "r") as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    cfg = om.merge(default_cfg, yaml_cfg, om.from_cli(args_list))
    cfg = DictConfig(cfg)
    return cfg
