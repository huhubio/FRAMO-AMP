"""Minimal YAML config loader with deep-merge over a default."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

PathLike = Union[str, Path]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: PathLike | None = None,
    *,
    default_path: PathLike | None = None,
) -> Dict[str, Any]:
    """Load a YAML config, optionally merged on top of a default config.

    Both files are YAML. ``path`` overrides values in ``default_path`` via a
    recursive dict merge. Either argument may be ``None``.
    """
    default: Dict[str, Any] = {}
    if default_path is not None and Path(default_path).is_file():
        with open(default_path, "r", encoding="utf-8") as fh:
            default = yaml.safe_load(fh) or {}

    if path is None:
        return default

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}

    return _deep_merge(default, user)
