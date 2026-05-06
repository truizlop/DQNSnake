"""Snake environment adapter for gym-snake.

This module wraps a single-player gym-snake env with a stable API used by the
bridge and future Swift integration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import random
from typing import Any

import numpy as np
from PIL import Image

try:
    import gym
except ImportError:  # pragma: no cover - environment dependent
    gym = None


@dataclass
class StepOutput:
    observation: np.ndarray
    reward: float
    done: bool
    score: float


class SnakeEnvAdapter:
    """Adapter around gym-snake that normalizes observation and API behavior."""

    def __init__(
        self,
        env_name: str = "Snake-v0",
        grid_size: int = 84,
        snake_dim: int = 20,
        tile_size: int | None = None,
        initial_length: int = 4,
        seed: int | None = None,
        normalize_rewards: bool = True,
        binary_observation: bool = True,
        alive_reward: float = 0.0005,
        potential_shaping_enabled: bool = False,
        potential_shaping_gamma: float = 0.99,
        potential_shaping_scale: float = 0.1,
    ):
        if gym is None:
            raise RuntimeError(
                "`gym` is required to use SnakeEnvAdapter. Install gym and gym-snake first."
            )
        self._ensure_snake_registered()
        self.grid_size = max(4, int(grid_size))
        self.snake_dim = max(4, int(snake_dim))
        if tile_size is None:
            # Keep rendered windows usable for larger grids.
            tile_size = max(4, 640 // self.snake_dim)
        self.tile_size = max(1, int(tile_size))

        self.env = self._make_env(
            env_name,
            env_kwargs={"dim": self.snake_dim, "size": self.tile_size},
        )
        self.initial_length = max(1, int(initial_length))
        self.seed = int(seed) if seed is not None else None
        self.normalize_rewards = bool(normalize_rewards)
        self.binary_observation = bool(binary_observation)
        self.alive_reward = float(alive_reward)
        self.potential_shaping_enabled = bool(potential_shaping_enabled)
        self.potential_shaping_gamma = float(potential_shaping_gamma)
        self.potential_shaping_scale = float(potential_shaping_scale)
        self.obs: np.ndarray | None = None
        self.done = False
        self.score = 0.0
        self._current_episode_frames: list[np.ndarray] = []
        self._last_episode_frames: list[np.ndarray] = []
        self._reset_count = 0
        self._seed_env_rngs()

    def reset(self) -> np.ndarray:
        if self._current_episode_frames:
            self._last_episode_frames = [frame.copy() for frame in self._current_episode_frames]
        reset_seed = self._next_reset_seed()
        raw = self._reset_with_seed(reset_seed)
        self.obs = self._extract_obs_from_reset(raw)
        self._enforce_initial_length()
        self.done = False
        self.score = 0.0
        grid = self._obs_to_grid(self.obs)
        self._current_episode_frames = [grid.copy()]
        return grid

    def step(self, action: int) -> tuple[np.ndarray, float, bool, float]:
        safe_action = self._sanitize_action(action)
        raw = self.env.step(safe_action)
        obs, raw_reward, done = self._extract_step_fields(raw)
        reward = self._transform_reward(raw_reward, done, prev_obs=self.obs, next_obs=obs)

        self.obs = obs
        self.done = bool(done)
        self.score += float(reward)

        output = StepOutput(
            observation=self._obs_to_grid(obs),
            reward=float(reward),
            done=self.done,
            score=float(self.score),
        )
        self._current_episode_frames.append(output.observation.copy())
        if self.done:
            self._last_episode_frames = [frame.copy() for frame in self._current_episode_frames]
        return output.observation, output.reward, output.done, output.score

    def save_last_episode_gif(
        self, path: str, scale: int = 8, frame_duration_ms: int = 80
    ) -> int:
        frames = self._last_episode_frames or self._current_episode_frames
        if not frames:
            raise RuntimeError("No episode frames available to export.")

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        scale = max(1, int(scale))
        frame_duration_ms = max(10, int(frame_duration_ms))

        pil_frames = []
        for grid in frames:
            rgb = self._grid_to_rgb(grid)
            image = Image.fromarray(rgb, mode="RGB")
            if scale > 1:
                image = image.resize(
                    (image.width * scale, image.height * scale),
                    Image.Resampling.NEAREST,
                )
            pil_frames.append(image)

        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
        return len(pil_frames)

    def _sanitize_action(self, action: int) -> int:
        unwrapped = getattr(self.env, "unwrapped", None)
        if unwrapped is None or not hasattr(unwrapped, "snake"):
            return int(action)
        return int(self._sanitize_action_for_snake(int(action), list(unwrapped.snake)))

    @staticmethod
    def _sanitize_action_for_snake(action: int, snake: list[list[int]]) -> int:
        # Actions in gym-snake: 0=left, 1=up, 2=right, 3=down.
        # If snake has a neck segment, forbid instant 180-degree turn into it.
        if len(snake) < 2:
            return action
        head_x, head_y = snake[0]
        neck_x, neck_y = snake[1]

        if head_x == neck_x:
            moving_right = head_y > neck_y
            if moving_right and action == 0:
                return 2
            if (not moving_right) and action == 2:
                return 0
            return action

        moving_down = head_x > neck_x
        if moving_down and action == 1:
            return 3
        if (not moving_down) and action == 3:
            return 1
        return action

    def get_frame(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("Environment has no frame yet. Call reset() first.")
        return self._obs_to_grid(self.obs)

    def get_score(self) -> float:
        return float(self.score)

    def is_done(self) -> bool:
        return bool(self.done)

    @staticmethod
    def _extract_obs_from_reset(raw: Any) -> Any:
        # gym <=0.25 returns obs; gymnasium/gym>=0.26 may return (obs, info)
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
            return raw[0]
        return raw

    @staticmethod
    def _extract_step_fields(raw: Any) -> tuple[Any, float, bool]:
        # gym <=0.25: (obs, reward, done, info)
        if isinstance(raw, tuple) and len(raw) == 4:
            obs, reward, done, _info = raw
            return obs, float(reward), bool(done)

        # gymnasium/gym>=0.26: (obs, reward, terminated, truncated, info)
        if isinstance(raw, tuple) and len(raw) == 5:
            obs, reward, terminated, truncated, _info = raw
            done = bool(terminated) or bool(truncated)
            return obs, float(reward), done

        raise ValueError(f"Unexpected step() output from environment: {type(raw)} {raw}")

    @staticmethod
    def _process_obs(obs: Any) -> np.ndarray:
        # Newer snake env variants may already return a grid.
        arr = np.array(obs, dtype=np.uint8)
        if arr.ndim == 2:
            return arr
        return arr

    @staticmethod
    def _ensure_snake_registered() -> None:
        # gym-snake registers its environments on import.
        try:
            import gym_snake  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "`gym-snake` is required. Install it with `pip install gym-snake`."
            ) from exc

    @staticmethod
    def _available_snake_envs() -> list[str]:
        return sorted(
            env_id
            for env_id in gym.envs.registry.keys()
            if "snake" in env_id.lower()
        )

    def _make_env(self, env_name: str, env_kwargs: dict[str, Any] | None = None):
        kwargs = env_kwargs or {}
        try:
            return gym.make(env_name, **kwargs)
        except Exception:
            if env_name != "Snake-v0":
                try:
                    return gym.make("Snake-v0", **kwargs)
                except Exception:
                    pass
            available = self._available_snake_envs()
            raise RuntimeError(
                f"Snake environment `{env_name}` is unavailable. "
                f"Registered snake envs: {available or 'none'}."
            )

    def _seed_env_rngs(self) -> None:
        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(self.seed)
        if hasattr(self.env, "observation_space") and hasattr(self.env.observation_space, "seed"):
            self.env.observation_space.seed(self.seed)

    def _next_reset_seed(self) -> int | None:
        if self.seed is None:
            return None
        seed = self.seed + self._reset_count
        self._reset_count += 1
        return seed

    def _reset_with_seed(self, seed: int | None) -> Any:
        if seed is None:
            return self.env.reset()
        try:
            return self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            return self.env.reset()

    def _enforce_initial_length(self) -> None:
        if self.initial_length <= 1:
            return

        unwrapped = getattr(self.env, "unwrapped", None)
        if unwrapped is None:
            return
        if not all(hasattr(unwrapped, name) for name in ("snake", "dim", "apple")):
            return
        if not unwrapped.snake:
            return

        head_x, head_y = unwrapped.snake[0]
        dim = int(unwrapped.dim)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(directions)

        for dx, dy in directions:
            candidate = [[head_x + i * dx, head_y + i * dy] for i in range(self.initial_length)]
            if all(0 <= x < dim and 0 <= y < dim for x, y in candidate):
                unwrapped.snake = candidate
                if hasattr(unwrapped, "new_apple"):
                    unwrapped.new_apple()
                return

    def _transform_reward(
        self,
        raw_reward: float,
        done: bool,
        prev_obs: Any | None = None,
        next_obs: Any | None = None,
    ) -> float:
        if not self.normalize_rewards:
            return float(raw_reward)

        if raw_reward >= 100:
            base_reward = 1.0
        elif done:
            base_reward = -1.0
        else:
            base_reward = self.alive_reward

        if not self.potential_shaping_enabled or prev_obs is None or next_obs is None:
            return base_reward

        prev_potential = self._state_potential(prev_obs)
        next_potential = self._state_potential(next_obs)
        shaping = self.potential_shaping_gamma * next_potential - prev_potential
        return float(base_reward + self.potential_shaping_scale * shaping)

    def _state_potential(self, obs: Any) -> float:
        head, apple = self._extract_head_and_apple(obs)
        if head is None or apple is None:
            return 0.0
        head_x, head_y = head
        apple_x, apple_y = apple
        distance = abs(head_x - apple_x) + abs(head_y - apple_y)
        max_distance = max(1, 2 * (self.snake_dim - 1))
        return -float(distance) / float(max_distance)

    def _extract_head_and_apple(
        self, obs: Any
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        if isinstance(obs, tuple) and len(obs) == 4:
            head_x, head_y, apple_x, apple_y = [int(v) for v in obs]
            return (head_x, head_y), (apple_x, apple_y)

        arr = np.array(obs)
        if arr.ndim == 2:
            head_candidates = np.argwhere(arr == 2)
            apple_candidates = np.argwhere(arr == 3)
            if len(head_candidates) > 0 and len(apple_candidates) > 0:
                hx, hy = head_candidates[0]
                ax, ay = apple_candidates[0]
                return (int(hx), int(hy)), (int(ax), int(ay))

        unwrapped = getattr(self.env, "unwrapped", None)
        if unwrapped is None:
            return None, None
        if hasattr(unwrapped, "snake") and hasattr(unwrapped, "apple") and unwrapped.snake:
            head = unwrapped.snake[0]
            apple = unwrapped.apple
            return (int(head[0]), int(head[1])), (int(apple[0]), int(apple[1]))

        return None, None

    def _obs_to_grid(self, obs: Any) -> np.ndarray:
        # gym-snake v0.1.7 returns a 4-value tuple (head_x, head_y, apple_x, apple_y).
        # Build the tile grid from environment state so downstream code gets image-like obs.
        if isinstance(obs, tuple) and len(obs) == 4 and hasattr(self.env, "unwrapped"):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "dim") and hasattr(unwrapped, "snake") and hasattr(unwrapped, "apple"):
                dim = int(unwrapped.dim)
                grid = np.zeros((dim, dim), dtype=np.uint8)

                snake = list(unwrapped.snake)
                for index, (x, y) in enumerate(snake):
                    if 0 <= x < dim and 0 <= y < dim:
                        # 1=body, 2=head
                        grid[x, y] = 2 if index == 0 else 1

                if len(unwrapped.apple) == 2:
                    ax, ay = int(unwrapped.apple[0]), int(unwrapped.apple[1])
                    if 0 <= ax < dim and 0 <= ay < dim:
                        # 3=apple
                        grid[ax, ay] = 3
                return self._resize_grid_if_needed(grid)

        arr = self._process_obs(obs)
        if self.binary_observation:
            arr = (arr > 0).astype(np.uint8)
        return self._resize_grid_if_needed(arr)

    def _resize_grid_if_needed(self, grid: np.ndarray) -> np.ndarray:
        if grid.ndim != 2 or grid.shape == (self.grid_size, self.grid_size):
            return grid
        src_h, src_w = grid.shape
        dst = self.grid_size
        row_idx = (np.arange(dst) * src_h / dst).astype(np.int64)
        col_idx = (np.arange(dst) * src_w / dst).astype(np.int64)
        return grid[row_idx][:, col_idx]

    @staticmethod
    def _grid_to_rgb(grid: np.ndarray) -> np.ndarray:
        rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        rgb[grid == 1] = np.array([30, 180, 30], dtype=np.uint8)  # body
        rgb[grid == 2] = np.array([120, 255, 120], dtype=np.uint8)  # head
        rgb[grid == 3] = np.array([220, 40, 40], dtype=np.uint8)  # apple
        return rgb


class FrameStack:
    """Simple frame stack utility for testing and preprocessing."""

    def __init__(self, k: int = 4):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = int(k)
        self.frames: deque[np.ndarray] = deque(maxlen=self.k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        frame_array = np.array(frame, dtype=np.uint8)
        for _ in range(self.k):
            self.frames.append(frame_array)
        return self._get()

    def step(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(np.array(frame, dtype=np.uint8))
        return self._get()

    def _get(self) -> np.ndarray:
        if len(self.frames) != self.k:
            raise RuntimeError("Frame stack is not initialized. Call reset(frame) first.")
        return np.stack(self.frames, axis=0)
