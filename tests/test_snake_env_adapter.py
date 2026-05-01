from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
PYTHON_DIR = os.path.join(ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

adapter_mod = importlib.import_module("snake_env_adapter")
SnakeEnvAdapter = adapter_mod.SnakeEnvAdapter
FrameStack = adapter_mod.FrameStack


def _has_gym_snake() -> bool:
    try:
        SnakeEnvAdapter()
        return True
    except Exception:
        return False


HAS_GYM_SNAKE = _has_gym_snake()


@pytest.mark.skipif(not HAS_GYM_SNAKE, reason="gym-snake environment unavailable")
def test_reset_and_frame_shape_consistency() -> None:
    env = SnakeEnvAdapter()
    obs = env.reset()

    assert obs.dtype == np.uint8
    assert obs.ndim == 2

    frame = env.get_frame()
    assert frame.shape == obs.shape
    assert np.array_equal(frame, obs)


@pytest.mark.skipif(not HAS_GYM_SNAKE, reason="gym-snake environment unavailable")
def test_step_returns_expected_contract() -> None:
    env = SnakeEnvAdapter()
    first = env.reset()

    obs, reward, done, score = env.step(int(env.env.action_space.sample()))

    assert obs.dtype == np.uint8
    assert obs.shape == first.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(score, float)
    assert score == pytest.approx(reward)


@pytest.mark.skipif(not HAS_GYM_SNAKE, reason="gym-snake environment unavailable")
def test_grid_contains_typed_entities() -> None:
    env = SnakeEnvAdapter(binary_observation=False)
    grid = env.reset()
    # 2=head must exist
    assert (grid == 2).any()
    # 3=apple must exist
    assert (grid == 3).any()


def test_frame_stack_shape() -> None:
    stack = FrameStack(k=4)
    obs = np.zeros((16, 16), dtype=np.uint8)
    stacked = stack.reset(obs)
    assert stacked.shape == (4, *obs.shape)

    obs2 = np.ones((16, 16), dtype=np.uint8)
    stacked2 = stack.step(obs2)

    assert stacked2.shape == (4, *obs.shape)
    assert np.array_equal(stacked2[-1], obs2)


def test_transform_reward_applies_alive_reward_for_non_terminal_step() -> None:
    adapter = object.__new__(SnakeEnvAdapter)
    adapter.normalize_rewards = True
    adapter.alive_reward = 0.001

    assert adapter._transform_reward(raw_reward=0.0, done=False) == pytest.approx(0.001)
    assert adapter._transform_reward(raw_reward=100.0, done=False) == pytest.approx(1.0)
    assert adapter._transform_reward(raw_reward=0.0, done=True) == pytest.approx(-1.0)
