from __future__ import annotations

import os
import time

from snake_env_adapter import SnakeEnvAdapter


def main() -> None:
    max_steps = int(os.environ.get("SNAKE_MAX_STEPS", "500"))
    fps = float(os.environ.get("SNAKE_FPS", "12"))
    frame_delay = 1.0 / fps if fps > 0 else 0.0
    control_mode = os.environ.get("SNAKE_CONTROL", "human").strip().lower()
    human_mode = control_mode == "human"

    env = SnakeEnvAdapter()
    env.reset()
    action = 2  # right

    try:
        for _ in range(max_steps):
            env.env.render(mode="human")
            if human_mode:
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            return
                        if event.key == pygame.K_r:
                            env.reset()
                            continue
                        if event.key == pygame.K_LEFT:
                            action = 0
                        elif event.key == pygame.K_UP:
                            action = 1
                        elif event.key == pygame.K_RIGHT:
                            action = 2
                        elif event.key == pygame.K_DOWN:
                            action = 3
            else:
                action = int(env.env.action_space.sample())

            _obs, _reward, done, _score = env.step(action)
            if frame_delay > 0:
                time.sleep(frame_delay)
            if done:
                env.reset()
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
