"""Configuration loading: TOML file + environment overrides.

`config.toml` holds every knob. Any value can be overridden with an environment
variable using a double-underscore path, which keeps secrets out of the file:

    BOTTRADE__PORTFOLIO__STARTING_CASH=5000
    BOTTRADE__SENTIMENT__MODEL=llm
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on ancient interpreters
    tomllib = None  # type: ignore[assignment]

ENV_PREFIX = "BOTTRADE__"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


class Config:
    """Thin wrapper over a nested dict with dotted-path lookups."""

    def __init__(self, data: dict[str, Any] | None = None, source: str = "<memory>"):
        self._data: dict[str, Any] = copy.deepcopy(data or {})
        self.source = source

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        data: dict[str, Any] = {}
        if path.exists():
            if tomllib is None:
                raise RuntimeError("TOML config requires Python 3.11+")
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        cfg = cls(data, source=str(path))
        cfg.apply_env_overrides()
        return cfg

    def apply_env_overrides(self) -> None:
        for env_key, raw in os.environ.items():
            if not env_key.startswith(ENV_PREFIX):
                continue
            path = env_key[len(ENV_PREFIX):].lower().split("__")
            self.set(".".join(path), _coerce(raw))

    # ----------------------------------------------------------------- access
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------- accessors
    def f(self, path: str, default: float = 0.0) -> float:
        try:
            return float(self.get(path, default))
        except (TypeError, ValueError):
            return default

    def i(self, path: str, default: int = 0) -> int:
        try:
            return int(self.get(path, default))
        except (TypeError, ValueError):
            return default

    def b(self, path: str, default: bool = False) -> bool:
        value = self.get(path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def s(self, path: str, default: str = "") -> str:
        value = self.get(path, default)
        return default if value is None else str(value)

    def list(self, path: str, default: list[Any] | None = None) -> list[Any]:
        value = self.get(path, default if default is not None else [])
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    # ------------------------------------------------------------------ misc
    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(source={self.source!r}, keys={list(self._data)})"


def _coerce(raw: str) -> Any:
    low = raw.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except (TypeError, ValueError):
            continue
    return raw
