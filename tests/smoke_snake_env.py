from __future__ import annotations

from snake_env_adapter import SnakeEnvAdapter


def main() -> None:
    env = SnakeEnvAdapter()
    obs = env.reset()
    print("reset shape:", obs.shape, "dtype:", obs.dtype)

    for i in range(10):
        action = env.env.action_space.sample()
        obs, reward, done, score = env.step(action)
        print(f"step={i} reward={reward:.2f} score={score:.2f} done={done} shape={obs.shape}")
        if done:
            break


if __name__ == "__main__":
    main()
