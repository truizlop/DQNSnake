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
                alive_reward = request.get("alive_reward", _env_float("SNAKE_ALIVE_REWARD", 0.0))
                with contextlib.redirect_stdout(sys.stderr):
                    env = SnakeEnvAdapter(env_name=env_name, seed=seed, alive_reward=alive_reward)
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

            if cmd == "quit":
                _ok()
                return 0

            _error(f"Unknown command: {cmd}")
        except Exception as exc:  # pragma: no cover - runtime dependent
            _error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
