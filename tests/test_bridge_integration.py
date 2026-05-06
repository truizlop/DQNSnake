from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from snake_env_adapter import SnakeEnvAdapter  # noqa: E402


def _start_bridge_server() -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(PYTHON_DIR)
    return subprocess.Popen(
        [sys.executable, str(PYTHON_DIR / "bridge_server.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

def _start_bridge_server_with_env(extra_env: dict[str, str]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(PYTHON_DIR)
    env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, str(PYTHON_DIR / "bridge_server.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _send(server: subprocess.Popen[str], payload: dict) -> dict:
    assert server.stdin is not None
    assert server.stdout is not None
    server.stdin.write(json.dumps(payload) + "\n")
    server.stdin.flush()
    line = server.stdout.readline()
    assert line, "Bridge server returned EOF unexpectedly."
    return json.loads(line)


def test_bridge_server_matches_adapter_for_seeded_action_sequence() -> None:
    seed = 777
    actions = [0, 1, 2, 3, 0, 0, 1, 2]

    adapter = SnakeEnvAdapter(seed=seed)
    adapter_frame = adapter.reset()

    server = _start_bridge_server()
    try:
        create_resp = _send(server, {"cmd": "create_env", "seed": seed})
        assert create_resp["ok"] is True

        reset_resp = _send(server, {"cmd": "reset"})
        assert reset_resp["ok"] is True
        server_frame = np.array(reset_resp["frame"], dtype=np.uint8)
        assert np.array_equal(server_frame, adapter_frame)

        for action in actions:
            adapter_obs, adapter_reward, adapter_done, adapter_score = adapter.step(action)
            step_resp = _send(server, {"cmd": "step", "action": action})
            assert step_resp["ok"] is True

            server_obs = np.array(step_resp["observation"], dtype=np.uint8)
            assert np.array_equal(server_obs, adapter_obs)
            assert float(step_resp["reward"]) == pytest.approx(adapter_reward)
            assert bool(step_resp["done"]) == adapter_done
            assert float(step_resp["score"]) == pytest.approx(adapter_score)
    finally:
        try:
            _send(server, {"cmd": "quit"})
        except Exception:
            pass
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_step_changes_observation_for_valid_motion() -> None:
    server = _start_bridge_server()
    try:
        assert _send(server, {"cmd": "create_env", "seed": 123})["ok"] is True
        frame0 = np.array(_send(server, {"cmd": "reset"})["frame"], dtype=np.uint8)

        # Try a few actions and expect at least one non-terminal move that changes the frame.
        changed = False
        for action in [0, 1, 2, 3]:
            resp = _send(server, {"cmd": "step", "action": action})
            assert resp["ok"] is True
            frame1 = np.array(resp["observation"], dtype=np.uint8)
            if not bool(resp["done"]) and not np.array_equal(frame0, frame1):
                changed = True
                break
        assert changed, "No action produced a changed non-terminal frame."
    finally:
        try:
            _send(server, {"cmd": "quit"})
        except Exception:
            pass
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_returns_error_for_unknown_command() -> None:
    server = _start_bridge_server()
    try:
        resp = _send(server, {"cmd": "does_not_exist"})
        assert resp["ok"] is False
        assert "Unknown command" in resp["error"]
    finally:
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_requires_create_env_before_step() -> None:
    server = _start_bridge_server()
    try:
        resp = _send(server, {"cmd": "step", "action": 0})
        assert resp["ok"] is False
        assert "Environment not created" in resp["error"]
    finally:
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_rejects_non_integer_action_payload() -> None:
    server = _start_bridge_server()
    try:
        assert _send(server, {"cmd": "create_env", "seed": 99})["ok"] is True
        resp = _send(server, {"cmd": "step", "action": "bad"})
        assert resp["ok"] is False
        assert "invalid literal" in resp["error"]
    finally:
        try:
            _send(server, {"cmd": "quit"})
        except Exception:
            pass
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_rejects_invalid_alive_reward_env_var() -> None:
    server = _start_bridge_server_with_env({"SNAKE_ALIVE_REWARD": "not-a-float"})
    try:
        resp = _send(server, {"cmd": "create_env", "seed": 99})
        assert resp["ok"] is False
        assert "SNAKE_ALIVE_REWARD" in resp["error"]
    finally:
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_rejects_invalid_potential_shaping_bool_env_var() -> None:
    server = _start_bridge_server_with_env({"SNAKE_POTENTIAL_SHAPING_ENABLE": "not-a-bool"})
    try:
        resp = _send(server, {"cmd": "create_env", "seed": 99})
        assert resp["ok"] is False
        assert "SNAKE_POTENTIAL_SHAPING_ENABLE" in resp["error"]
    finally:
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_supports_configured_env_dim_and_grid_size() -> None:
    server = _start_bridge_server_with_env({"SNAKE_ENV_DIM": "20", "SNAKE_GRID_SIZE": "84"})
    try:
        assert _send(server, {"cmd": "create_env", "seed": 7})["ok"] is True
        reset = _send(server, {"cmd": "reset"})
        assert reset["ok"] is True
        frame = np.array(reset["frame"], dtype=np.uint8)
        assert frame.shape == (84, 84)
    finally:
        try:
            _send(server, {"cmd": "quit"})
        except Exception:
            pass
        server.kill()
        server.wait(timeout=5)


def test_bridge_server_can_save_last_episode_gif() -> None:
    import tempfile

    server = _start_bridge_server()
    try:
        assert _send(server, {"cmd": "create_env", "seed": 7})["ok"] is True
        assert _send(server, {"cmd": "reset"})["ok"] is True
        _send(server, {"cmd": "step", "action": 0})
        _send(server, {"cmd": "step", "action": 1})

        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "best.gif")
            resp = _send(
                server,
                {"cmd": "save_last_episode_gif", "path": path, "scale": 2, "frame_duration_ms": 40},
            )
            assert resp["ok"] is True
            assert int(resp["frame_count"]) >= 1
            assert Path(path).exists()
    finally:
        try:
            _send(server, {"cmd": "quit"})
        except Exception:
            pass
        server.kill()
        server.wait(timeout=5)
