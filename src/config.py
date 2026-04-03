"""
config.py
=========
Loads project configuration from config.yaml.
"""
from __future__ import annotations

import pathlib
from typing import Any


def load_config(path: str | pathlib.Path = "config.yaml") -> dict[str, Any]:
    """Load and return the YAML config as a plain dict.

    Parameters
    ----------
    path:
        Path to the YAML file.  Defaults to ``config.yaml`` in the current
        working directory (i.e. the repo root when running notebooks).

    Returns
    -------
    dict
        Parsed config, or an empty dict if the file does not exist.
    """
    import yaml  # soft dependency — already required by project

    p = pathlib.Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}
