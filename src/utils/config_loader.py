from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """LOAD YAML CONFIG FILE. **"""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("Config file is empty.")

    return config


def get_enabled_experiments(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """RETURN ENABLED EXPERIMENTS FROM CONFIG. **"""
    experiments = config.get("experiments", {})

    return {
        name: settings
        for name, settings in experiments.items()
        if settings.get("enabled", False)
    }