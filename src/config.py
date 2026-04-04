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
    p = pathlib.Path(path)
    if not p.exists():
        return {}

    try:
        import yaml  # optional dependency used only when parsing YAML files
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load configuration from "
            f"{p!s}. Install it with `pip install PyYAML`."
        ) from exc
    with p.open() as fh:
        return yaml.safe_load(fh) or {}
