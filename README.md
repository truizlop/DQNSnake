# DQNSnake

Single-player Snake environment setup with a Swift bridge, ready for DQN work.

## What is implemented

- Python snake environment adapter around `gym-snake`
- Python bridge surface (`create_env`, `reset`, `step`, `get_frame`, `get_score`, `is_done`)
- Swift `SnakeEnv` actor API controlling the Python environment through a JSON subprocess bridge
- Frame stacking on both Python and Swift sides
- Visual gameplay runners (Python-controlled and Swift-controlled)
- MLX training scaffold target (`snake-dqn-train`) for upcoming DQN implementation

## Repository structure

- `python/`
  - `snake_env_adapter.py`: normalized env API, score tracking, frame grid conversion, reward mapping, initial snake length setup
  - `bridge.py`: simple callable bridge module
  - `bridge_server.py`: line-delimited JSON server used by Swift bridge
- `swift/`
  - `Sources/SnakeEnv/`: actor API + bridge client
  - `Sources/SnakeEnvCLI/`: Swift runner for environment stepping
  - `Sources/SnakeDQNTrain/`: DQN + MLX scaffold target
- `tests/`
  - `smoke_snake_env.py`: smoke runner
  - `visual_snake_env.py`: pygame visual runner
  - `test_snake_env_adapter.py`: adapter and frame stack tests

## Data/reward contract

- Observation is exposed as a 2D grid (`UInt8` / `np.uint8`) where:
  - `0` = empty
  - `1` = snake body
  - `2` = snake head
  - `3` = apple
- Rewards are normalized to:
  - `+1` apple
  - `0` normal step
  - `-1` terminal collision
- Episode score is cumulative reward.

## Run commands

From repo root:

- `make setup`: install Python deps (`gym`, `gym-snake`, `numpy`, `pytest`)
- `make build`: build Swift package
- `make test`: run Python + Swift tests
- `make run`: run Python smoke script
- `make run-visual`: Python-controlled pygame visual loop
  - `SNAKE_CONTROL=human make run-visual` for keyboard control (arrows, `R`, `Esc`/`Q`)
- `make run-swift`: Swift-controlled rollout via bridge
- `make run-swift-visual`: Swift-controlled rollout + pygame rendering
- `make run-dqn`: run DQN scaffold target

Useful env vars:

- `SNAKE_STEPS` (default `30` for `run-swift`, `20` for `run-dqn`)
- `SNAKE_FPS` (visual modes)
- `SNAKE_MAX_STEPS` (visual Python loop)
- `SNAKE_PYTHON_EXE` (Python binary for Swift bridge; defaults to `/opt/anaconda3/bin/python3` when available)
- `SNAKE_PYTHON_DIR` (Python module directory; defaults to `python/`)
- `SNAKE_ENABLE_MLX=1` to enable MLX tensor path in `run-dqn`

## Swift API

`SnakeEnv` actor currently exposes:

- `reset() async throws -> Frame`
- `step(action: Int) async throws -> StepResult`
- `score() async -> Float`
- `isDone() async -> Bool`
- `render() async throws`

## Current caveats

- `gym-snake` upstream uses old Gym APIs and emits deprecation warnings.
- Reversing direction into the snake body causes immediate terminal state (confirmed behavior).
- `mlx-swift` command-line builds may fail to run GPU/Metal shader paths in pure SwiftPM CLI workflows.
  - The DQN scaffold defaults to env-only mode unless `SNAKE_ENABLE_MLX=1`.
  - For full MLX GPU workflows, prefer Xcode/xcodebuild integration.

## Next step

Implement DQN in `swift/Sources/SnakeDQNTrain/`:

- Q-network (`MLXNN`)
- replay buffer
- epsilon-greedy policy
- target network updates
- optimizer/loss training step (`MLXOptimizers`)
