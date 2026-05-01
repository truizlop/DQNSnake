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
