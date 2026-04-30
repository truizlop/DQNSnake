# DQNSnake

Single-player Snake training stack with:
- Python `gym-snake` environment
- Swift `SnakeEnv` bridge actor
- Swift DQN training loop (MLX/MLXNN/MLXOptimizers)
- Checkpointing, resume, evaluation, and TensorBoard metric streaming

## Current capabilities

- Python snake adapter around `gym-snake`
- Typed Swift↔Python JSON bridge protocol payloads
- Swift `SnakeEnv` actor API
- DQN components in Swift:
  - replay buffer (explicit sampling strategy)
  - epsilon-greedy exploration policy
  - online/target network training
  - periodic target sync
  - checkpoint save/load (`.safetensors`)
  - periodic greedy evaluation loop
  - training quality checks (loss/Q scale/gradient norm + skip-on-invalid update)
- TensorBoard integration from Swift training process

## Repository structure

- `python/`
  - `snake_env_adapter.py`: normalized env API, score tracking, frame grid conversion, reward mapping, initial snake length setup
  - `bridge.py`: simple callable bridge module
  - `bridge_server.py`: line-delimited JSON server used by Swift bridge
  - `tensorboard_stream.py`: JSON scalar stream -> TensorBoard event files
- `swift/`
  - `Sources/SnakeEnv/`: actor API + bridge client
  - `Sources/SnakeEnvCLI/`: Swift runner for environment stepping
  - `Sources/SnakeDQNTrain/`: DQN training pipeline
- `tests/`
  - `smoke_snake_env.py`: smoke runner
  - `visual_snake_env.py`: pygame visual runner
  - `test_snake_env_adapter.py`: adapter and frame stack tests
  - `swift/Tests/SnakeDQNTrainTests/`: DQN unit tests (checkpoint manager, policy behavior)

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
- `score` returned by env is cumulative reward.

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
- `make run-dqn`: run Swift DQN trainer target

Useful env vars:

- `SNAKE_STEPS` (default `30` for `run-swift`, `20` for `run-dqn`)
- `SNAKE_FPS` (visual modes)
- `SNAKE_MAX_STEPS` (visual Python loop)
- `SNAKE_MAX_EPISODE_STEPS` (Swift DQN trainer)
- `SNAKE_PYTHON_EXE` (Python binary for Swift bridge; defaults to `/opt/anaconda3/bin/python3` when available)
- `SNAKE_PYTHON_DIR` (Python module directory; defaults to `python/`)
- `SNAKE_RESUME_CHECKPOINT` (path to `.safetensors` checkpoint to resume from)

Evaluation env vars:
- `SNAKE_EVAL_EVERY_EPISODES` (default `0`, disabled)
- `SNAKE_EVAL_EPISODES` (default `5`)

TensorBoard env vars:
- `SNAKE_TB_ENABLE` (`1` default)
- `SNAKE_TB_LAUNCH` (`1` default; auto-launches TensorBoard process)
- `SNAKE_TB_LOGDIR` (default `runs/snake_dqn`)
- `SNAKE_TB_PORT` (default `6006`)

## TensorBoard usage

`snake-dqn-train` can publish metrics directly to TensorBoard event files through `python/tensorboard_stream.py`.

Install TensorBoard in your Python environment:
- `python3 -m pip install tensorboard`

Typical run:
- `make run-dqn`

If `SNAKE_TB_ENABLE=1` and `SNAKE_TB_LAUNCH=1`, TensorBoard is started automatically and available at:
- `http://localhost:6006` (or `SNAKE_TB_PORT`)

Logged scalar groups:
- `train/loss`
- `train/q_mean_abs`
- `train/q_max_abs`
- `train/grad_l2`
- `train/skipped_update`
- `train/episode_reward`
- `train/episode_score`
- `train/episode_steps`
- `eval/avg_reward`
- `eval/avg_score`

## Swift API

`SnakeEnv` actor currently exposes:

- `reset() async throws -> Frame`
- `step(action: Int) async throws -> StepResult`
- `score() async -> Float`
- `isDone() async -> Bool`
- `render() async throws`

## Training behavior summary

- Replay buffer sampling is explicit:
  - `withReplacement`
  - `withoutReplacement`
- Checkpointing:
  - periodic: `model_step_<globalStep>.safetensors`
  - best: `model_best.safetensors` (currently best by training episode score)
- Resume:
  - loads online model from checkpoint
  - syncs target model from loaded online model
  - resumes `global_step` from checkpoint metadata when present
- Update quality checks:
  - tracks loss, Q scale, gradient L2 norm
  - skips optimizer update if loss/gradients are non-finite

## Current caveats / notes

- `gym-snake` upstream uses old Gym APIs and emits deprecation warnings.
- Reversing direction into the snake body causes immediate terminal state (confirmed behavior).
- MLX runtime requirements (Metal / bundled libs) still apply depending on your local setup.
- Optimizer state checkpoint/resume is not yet implemented (model weights resume is implemented).
