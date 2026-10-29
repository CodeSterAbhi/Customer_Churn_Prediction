import os
import yaml
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config