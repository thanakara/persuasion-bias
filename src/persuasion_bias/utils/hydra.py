import logging.config

import yaml

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig, read_write
from hydra.core.hydra_config import HydraConfig

from persuasion_bias.utils import CONFIG_DIR, decide_device
from persuasion_bias.prompts import templates

_LOG_CONFIG = CONFIG_DIR / "hydra" / "job_logging" / "custom.yaml"


def register_resolvers():
    OmegaConf.register_new_resolver("decide_device", decide_device)
    OmegaConf.register_new_resolver("prompt", lambda name: getattr(templates, name))


def setup_logging() -> None:
    with open(_LOG_CONFIG) as f:
        logging.config.dictConfig(yaml.safe_load(f))


def load_config(overrides: list[str] | None = None, setup_log: bool = True) -> DictConfig:
    if setup_log:
        setup_logging()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [], return_hydra_config=True)

    HydraConfig.instance().set_config(cfg)
    with read_write(cfg["hydra"]):
        OmegaConf.resolve(cfg)

    return cfg
