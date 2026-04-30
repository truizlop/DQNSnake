"""Minimal Python bridge surface intended for Swift interop layers."""

from __future__ import annotations

from threading import Lock
from typing import Optional

import numpy as np

from snake_env_adapter import SnakeEnvAdapter

_env: Optional[SnakeEnvAdapter] = None
_lock = Lock()
_last_error: Optional[str] = None


def create_env(env_name: str = "Snake-v0", seed: int | None = None) -> int:
    global _env
    global _last_error
    with _lock:
        try:
            _env = SnakeEnvAdapter(env_name=env_name, seed=seed)
            _last_error = None
            return 1
        except Exception as exc:  # pragma: no cover - runtime dependent
            _env = None
            _last_error = str(exc)
            return 0


def _require_env() -> SnakeEnvAdapter:
    if _env is None:
        raise RuntimeError("Environment not created. Call create_env() first.")
    return _env


def reset() -> np.ndarray:
    with _lock:
        return _require_env().reset()


def step(action: int) -> tuple[np.ndarray, float, bool, float]:
    with _lock:
        return _require_env().step(action)


def get_frame() -> np.ndarray:
    with _lock:
        return _require_env().get_frame()


def get_score() -> float:
    with _lock:
        return _require_env().get_score()


def is_done() -> bool:
    with _lock:
        return _require_env().is_done()


def get_last_error() -> str:
    with _lock:
        return _last_error or ""
