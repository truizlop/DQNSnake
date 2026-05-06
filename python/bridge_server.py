"""Line-delimited JSON bridge server for Swift <-> Python Snake integration."""

from __future__ import annotations

import json
import contextlib
import os
import sys
from typing import Any

from snake_env_adapter import SnakeEnvAdapter


env: SnakeEnvAdapter | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {raw}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean for {name}: {raw}")


def _ok(**payload: Any) -> None:
    payload["ok"] = True
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _error(message: str) -> None:
    sys.stdout.write(json.dumps({"ok": False, "error": message}) + "\n")
    sys.stdout.flush()


def _require_env() -> SnakeEnvAdapter:
    if env is None:
        raise RuntimeError("Environment not created. Call create_env first.")
    return env


def main() -> int:
    global env

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            request = json.loads(raw)
            cmd = request.get("cmd")

            if cmd == "create_env":
                env_name = request.get("env_name", "Snake-v0")
                seed = request.get("seed")
                alive_reward = request.get("alive_reward", _env_float("SNAKE_ALIVE_REWARD", 0.0005))
                potential_shaping_enabled = request.get(
                    "potential_shaping_enabled",
                    _env_bool("SNAKE_POTENTIAL_SHAPING_ENABLE", False),
                )
                potential_shaping_gamma = request.get(
                    "potential_shaping_gamma",
                    _env_float("SNAKE_POTENTIAL_SHAPING_GAMMA", 0.99),
                )
                potential_shaping_scale = request.get(
                    "potential_shaping_scale",
                    _env_float("SNAKE_POTENTIAL_SHAPING_SCALE", 0.1),
                )
                snake_dim = int(request.get("snake_dim", os.getenv("SNAKE_ENV_DIM", "20")))
                grid_size = int(request.get("grid_size", os.getenv("SNAKE_GRID_SIZE", "84")))
                with contextlib.redirect_stdout(sys.stderr):
                    env = SnakeEnvAdapter(
                        env_name=env_name,
                        seed=seed,
                        alive_reward=alive_reward,
                        potential_shaping_enabled=potential_shaping_enabled,
                        potential_shaping_gamma=potential_shaping_gamma,
                        potential_shaping_scale=potential_shaping_scale,
                        snake_dim=snake_dim,
                        grid_size=grid_size,
                    )
                _ok()
                continue

            if cmd == "reset":
                with contextlib.redirect_stdout(sys.stderr):
                    frame = _require_env().reset().tolist()
                _ok(frame=frame)
                continue

            if cmd == "step":
                action = int(request["action"])
                with contextlib.redirect_stdout(sys.stderr):
                    obs, reward, done, score = _require_env().step(action)
                _ok(observation=obs.tolist(), reward=reward, done=done, score=score)
                continue

            if cmd == "score":
                with contextlib.redirect_stdout(sys.stderr):
                    score = _require_env().get_score()
                _ok(score=score)
                continue

            if cmd == "is_done":
                with contextlib.redirect_stdout(sys.stderr):
                    done = _require_env().is_done()
                _ok(done=done)
                continue

            if cmd == "render":
                with contextlib.redirect_stdout(sys.stderr):
                    _require_env().env.render(mode="human")
                _ok()
                continue

            if cmd == "save_last_episode_gif":
                path = request["path"]
                scale = int(request.get("scale", 8))
                frame_duration_ms = int(request.get("frame_duration_ms", 80))
                with contextlib.redirect_stdout(sys.stderr):
                    frame_count = _require_env().save_last_episode_gif(
                        path=path,
                        scale=scale,
                        frame_duration_ms=frame_duration_ms,
                    )
                _ok(frame_count=frame_count)
                continue

            if cmd == "quit":
                _ok()
                return 0

            _error(f"Unknown command: {cmd}")
        except Exception as exc:  # pragma: no cover - runtime dependent
            _error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
