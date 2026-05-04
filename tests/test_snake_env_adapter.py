from __future__ import annotations

import importlib
import os
import sys
import tempfile

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


def test_resize_grid_if_needed_maps_to_target_shape() -> None:
    adapter = object.__new__(SnakeEnvAdapter)
    adapter.grid_size = 84
    small = np.zeros((20, 20), dtype=np.uint8)
    small[3, 4] = 2
    out = adapter._resize_grid_if_needed(small)
    assert out.shape == (84, 84)
    assert (out == 2).any()


def test_sanitize_action_for_snake_blocks_reverse_turn() -> None:
    # Moving right (head at larger y than neck), reverse-left should become right.
    snake_right = [[5, 6], [5, 5], [5, 4]]
    assert SnakeEnvAdapter._sanitize_action_for_snake(0, snake_right) == 2
    assert SnakeEnvAdapter._sanitize_action_for_snake(1, snake_right) == 1

    # Moving up (head at smaller x than neck), reverse-down should become up.
    snake_up = [[4, 5], [5, 5], [6, 5]]
    assert SnakeEnvAdapter._sanitize_action_for_snake(3, snake_up) == 1
    assert SnakeEnvAdapter._sanitize_action_for_snake(0, snake_up) == 0


def test_save_last_episode_gif_writes_file() -> None:
    adapter = object.__new__(SnakeEnvAdapter)
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2, 3] = 2
    frame[5, 6] = 3
    adapter._current_episode_frames = [frame.copy(), frame.copy()]
    adapter._last_episode_frames = []

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "episode.gif")
        frame_count = adapter.save_last_episode_gif(path=path, scale=2, frame_duration_ms=40)
        assert frame_count == 2
        assert os.path.exists(path)
