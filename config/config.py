from typing import List, Optional, Type

from omegaconf import OmegaConf

from config.schema import ExperimentConfig


def load_config(path: str, overrides: Optional[List[str]] = None, schema: Type = ExperimentConfig):
    base = OmegaConf.structured(schema)
    yaml_cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(base, yaml_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_readonly(cfg, True)
    return cfg

