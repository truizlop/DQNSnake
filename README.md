# DQNSnake

DQNSnake is a hybrid Snake reinforcement-learning project:

- Python owns the `gym-snake` environment adapter, rendering, GIF export, and JSON bridge server.
- Swift owns the environment actor, typed bridge client, DQN implementation, replay buffer, training loop, evaluation loop, checkpointing, and model playback.
- MLX/MLXNN/MLXOptimizers provide the neural network, optimizer, tensor operations, and `.safetensors` checkpoint format.

The current best checkpoint is expected at `checkpoints/model_best.safetensors`. The curated training history and qualitative GIF progression live in [training.md](training.md).

## Quick Start

Install Python dependencies:

```sh
make setup
```

Run all tests:

```sh
make test
```

Play Snake with the best trained model:

```sh
make play-dqn
```

Play with live neural-network activation heatmaps:

```sh
SNAKE_PLAY_ACTIVATIONS=1 \
SNAKE_PLAY_ACTIVATION_DASHBOARD=1 \
SNAKE_PLAY_STEP_MODE=1 \
make play-dqn
```

Evaluate the best checkpoint greedily:

```sh
make eval-dqn
```

Start a training run with the default Swift baseline:

```sh
make train-dqn
```

Run a long GPU continuation with the latest strong setup:

```sh
SNAKE_MLX_DEVICE=gpu \
SNAKE_STEPS=25000000 \
SNAKE_RESUME_CHECKPOINT=checkpoints/model_step_14990000.safetensors \
SNAKE_DQN_ALGORITHM=double \
SNAKE_REPLAY_SAMPLING_STRATEGY=prioritized \
SNAKE_PER_ALPHA=0.6 \
SNAKE_PER_BETA_START=0.4 \
SNAKE_PER_BETA_ANNEAL_STEPS=1000000 \
SNAKE_PER_EPSILON=0.001 \
SNAKE_EPSILON_END=0.05 \
SNAKE_LEARNING_RATE=0.00025 \
SNAKE_LEARNING_RATE_FINAL=0.000125 \
SNAKE_LEARNING_RATE_DECAY_START_STEP=15000000 \
SNAKE_LEARNING_RATE_DECAY_END_STEP=25000000 \
make train-dqn
```

## Make Targets

Current primary targets:

- `make help`: show available commands.
- `make docs`: print the Makefile command documentation header.
- `make setup`: install Python runtime and test dependencies.
- `make build`: build the Swift package.
- `make test`: run Python and Swift tests.
- `make smoke-python`: run the Python environment smoke test.
- `make visual-python`: run the Python visual loop; use `SNAKE_CONTROL=human` for keyboard control.
- `make smoke-swift`: run the Swift bridge CLI with random actions.
- `make visual-swift`: run the Swift bridge CLI with pygame rendering.
- `make train-dqn`: build prerequisites, prepare MLX metallib, then run Swift DQN training.
- `make eval-dqn`: build prerequisites, prepare MLX metallib, then run greedy evaluation against a checkpoint.
- `make play-dqn`: build prerequisites, prepare MLX metallib, then run model-controlled Snake playback and export a GIF.
- `make open-xcode`: open the Swift package in Xcode with a sanitized Apple toolchain environment.
- `make clean`: remove local build/test caches while keeping `runs/` and `checkpoints/`.

Backward-compatible aliases remain for older commands:

- `make run` and `make run-smoke` -> `make smoke-python`
- `make run-visual` -> `make visual-python`
- `make run-swift` -> `make smoke-swift`
- `make run-swift-visual` -> `make visual-swift`
- `make run-dqn` -> `make train-dqn`

## Repository Layout

- `python/snake_env_adapter.py`: wraps `gym-snake`, normalizes rewards/observations, handles action sanitization, ignores the legacy 200-step upstream time-limit terminal, and exports episode GIFs.
- `python/bridge_server.py`: line-delimited JSON bridge used by Swift.
- `python/bridge.py`: direct Python bridge helpers.
- `python/tensorboard_stream.py`: writes scalar events consumed by TensorBoard.
- `python/activation_viewer.py`: live pygame dashboard for neural-network activation heatmaps during model playback.
- `swift/Sources/SnakeEnv/`: Swift actor API, frame stack, typed bridge client, and bridge payloads.
- `swift/Sources/SnakeEnvCLI/`: simple Swift rollout CLI for bridge smoke testing.
- `swift/Sources/SnakeDQNTrain/`: DQN model, learner, trainer, replay buffer, evaluator, player, checkpointing, metrics, schedules, and runtime parsing.
- `tests/`: Python adapter and bridge integration tests.
- `swift/Tests/`: Swift unit and integration tests for bridge behavior, replay, learner math/stability, parser defaults, schedules, logging, and tensor layout.
- `runs/`: generated local TensorBoard/observability run artifacts; not required for the committed project record.
- `checkpoints/`: retained meaningful `.safetensors` checkpoints.
- `training_gifs/`: curated GIFs illustrating qualitative learning progression.

## Architecture

The training process is launched from Swift through the `snake-dqn-train` executable.

1. Swift starts or connects to the Python bridge.
2. Python creates a normalized `gym-snake` environment.
3. Swift resets the environment and maintains a 4-frame stack.
4. The stacked frames are converted to an MLX tensor shaped `[1, height, width, 4]`.
5. The online DQN chooses actions through epsilon-greedy exploration during training.
6. Transitions are stored in replay.
7. After warmup, Swift samples minibatches and trains the online network.
8. The target network is periodically synchronized from the online network.
9. Evaluation runs greedily, without epsilon exploration.
10. Checkpoints, TensorBoard metrics, structured logs, and best-episode GIFs are written during training.
11. Play mode can optionally inspect the network and stream activations to a separate pygame dashboard.

## Observation Contract

The model input is intentionally image-like and binary:

- Python reconstructs a grid from `gym-snake` state.
- By default, the training observation is binary: `0` for empty and `1` for any occupied/meaningful tile.
- The agent receives 4 stacked frames so the network can infer movement, head position, and dynamics from temporal changes instead of explicit entity labels.
- The Python adapter keeps a typed grid internally for rendering/GIF export so apples, head, and body can be colored correctly in visual output.
- The default environment board is `20x20`, resized by nearest-neighbor sampling to `84x84` for the model.

This was a deliberate decision: the input resembles a binarized screenshot rather than a hand-authored feature vector.

## Reward Contract

Normalized rewards are:

- `+1` for eating an apple.
- `SNAKE_ALIVE_REWARD` for a normal non-terminal step, default `0.0005`.
- `-1` for terminal collision.

Optional potential-based shaping is available but not part of the latest strong baseline:

- `SNAKE_POTENTIAL_SHAPING_ENABLE=1`
- `SNAKE_POTENTIAL_SHAPING_GAMMA=0.99`
- `SNAKE_POTENTIAL_SHAPING_SCALE=0.1`

The shaping term is based on Manhattan distance to the apple and is implemented as `gamma * Phi(s') - Phi(s)`, so it can provide denser directional feedback while preserving the optimal policy under the usual potential-based shaping assumptions.

## Current DQN Implementation

Implemented training features:

- Int-backed `SnakeAction` enum to prevent illegal action values in DQN transitions.
- 4-frame state stacking.
- Convolutional DQN model in Swift/MLX.
- Epsilon-greedy exploration policy.
- Replay buffer with explicit sampling strategy.
- Uniform sampling with and without replacement.
- Prioritized Experience Replay with alpha, beta annealing, epsilon, importance weights, and TD-error priority updates.
- Single DQN and Double DQN modes through `SNAKE_DQN_ALGORITHM`.
- Online and target networks with periodic target synchronization.
- Bellman TD targets with `gamma=0.99` by default.
- Gradient clipping.
- Loss, Q-scale, gradient-norm, and non-finite update guards.
- Learning-rate decay controls.
- Greedy evaluation loop.
- Evaluation-driven best checkpoint selection.
- Checkpoint resume for model weights and global-step metadata.
- TensorBoard scalar metrics.
- Structured JSONL observability logs.
- Resource telemetry.
- Best-training-episode GIF capture.
- Play-only mode for running a trained checkpoint.

Optimizer state is not checkpointed. Resume restores model weights and global-step metadata, then synchronizes the target network from the loaded online network.

## Baseline Defaults

The named baseline lives in `swift/Sources/SnakeDQNTrain/DQNHyperparameterBaseline.swift` as `snakeV1`.

Important defaults:

- `totalEnvironmentSteps`: `200_000`
- `maxStepsPerEpisode`: `2_000`
- `replayBufferCapacity`: `100_000`
- `replaySamplingStrategy`: `withReplacement`
- `warmupSteps`: `5_000`
- `trainEvery`: `4`
- `targetSyncEvery`: `10_000`
- `batchSize`: `32`
- `gamma`: `0.99`
- `learningRate`: `0.00025`
- `gradientClipNorm`: `10`
- `dqnAlgorithm`: `double`
- `epsilonStart`: `1.0`
- `epsilonEnd`: `0.1`
- `epsilonDecaySteps`: `100_000`
- `evalEveryEpisodes`: `25`
- `evalEpisodes`: `5`
- `prioritizedReplayAlpha`: `0.6`
- `prioritizedReplayBetaStart`: `0.4`
- `prioritizedReplayBetaAnnealSteps`: `1_000_000`
- `prioritizedReplayEpsilon`: `0.001`

The default sampling strategy remains uniform with replacement for simple runs. Long training runs should usually override it with `SNAKE_REPLAY_SAMPLING_STRATEGY=prioritized`.

## Runtime Configuration

Core training variables:

- `SNAKE_STEPS`: total environment steps.
- `SNAKE_MAX_EPISODE_STEPS`: Swift-side episode cap.
- `SNAKE_RESUME_CHECKPOINT`: checkpoint path to load.
- `SNAKE_MLX_DEVICE`: `cpu` or `gpu`.
- `SNAKE_SEED`: deterministic Swift replay/exploration RNG plus Python reset seeding.
- `SNAKE_DQN_ALGORITHM`: `single` or `double`.
- `SNAKE_REPLAY_SAMPLING_STRATEGY`: `with_replacement`, `without_replacement`, or `prioritized`.
- `SNAKE_BATCH_SIZE`: minibatch size.
- `SNAKE_WARMUP_STEPS`: replay warmup before training.
- `SNAKE_TRAIN_EVERY`: optimize every N environment steps.
- `SNAKE_TARGET_SYNC_EVERY`: target-network sync interval.
- `SNAKE_CHECKPOINT_EVERY_STEPS`: periodic checkpoint interval.
- `SNAKE_CHECKPOINT_DIR`: checkpoint output directory.

PER variables:

- `SNAKE_PER_ALPHA`: prioritization exponent.
- `SNAKE_PER_BETA_START`: initial importance-sampling correction.
- `SNAKE_PER_BETA_ANNEAL_STEPS`: steps to anneal beta to `1.0`.
- `SNAKE_PER_EPSILON`: small constant added to TD error before priority calculation.

Exploration and learning-rate variables:

- `SNAKE_EPSILON_START`: initial epsilon.
- `SNAKE_EPSILON_END`: epsilon floor.
- `SNAKE_EPSILON_DECAY_STEPS`: linear epsilon decay duration.
- `SNAKE_LEARNING_RATE`: initial learning rate.
- `SNAKE_LEARNING_RATE_FINAL`: optional final learning rate.
- `SNAKE_LEARNING_RATE_DECAY_START_STEP`: global step where LR decay starts.
- `SNAKE_LEARNING_RATE_DECAY_END_STEP`: global step where LR decay ends.
- `SNAKE_GRAD_CLIP_NORM`: gradient clipping threshold.

Environment variables:

- `SNAKE_PYTHON_EXE`: Python binary used by Swift bridge.
- `SNAKE_PYTHON_DIR`: Python module directory, usually `python`.
- `SNAKE_ENV_DIM`: underlying Snake board dimension, default `20`.
- `SNAKE_GRID_SIZE`: observation size after resize, default `84`.
- `SNAKE_ALIVE_REWARD`: alive-step reward, default `0.0005`.
- `SNAKE_POTENTIAL_SHAPING_ENABLE`: enable potential-based shaping.
- `SNAKE_POTENTIAL_SHAPING_GAMMA`: shaping gamma.
- `SNAKE_POTENTIAL_SHAPING_SCALE`: shaping multiplier.

Evaluation variables:

- `SNAKE_EVAL_ONLY`: run evaluation only.
- `SNAKE_EVAL_EVERY_EPISODES`: training evaluation frequency.
- `SNAKE_EVAL_EPISODES`: number of greedy eval episodes.
- `SNAKE_EVAL_SEEDS`: comma-separated fixed eval seeds.
- `SNAKE_EVAL_ROLLING_WINDOW`: moving-average window in evaluation summaries.

TensorBoard and observability variables:

- `SNAKE_TB_ENABLE`: enable TensorBoard metrics.
- `SNAKE_TB_LAUNCH`: launch TensorBoard automatically.
- `SNAKE_TB_LOGDIR`: TensorBoard log directory.
- `SNAKE_TB_PORT`: TensorBoard port.
- `SNAKE_OBS_ENABLE`: enable structured JSONL logs.
- `SNAKE_OBS_LOG_PATH`: structured log path.
- `SNAKE_RESOURCE_TELEMETRY_EVERY_STEPS`: resource telemetry interval, or `0` to disable.

GIF and play variables:

- `SNAKE_BEST_GIF_ENABLE`: enable best-training-episode GIF capture.
- `SNAKE_BEST_GIF_DIR`: best-training-episode GIF directory.
- `SNAKE_BEST_GIF_SCALE`: GIF pixel scale.
- `SNAKE_BEST_GIF_FRAME_MS`: GIF frame duration.
- `SNAKE_PLAY_ONLY`: run model playback only.
- `SNAKE_PLAY_EPISODES`: number of playback episodes.
- `SNAKE_PLAY_RENDER`: render playback through Python/pygame.
- `SNAKE_PLAY_GIF_DIR`: playback GIF output directory.
- `SNAKE_PLAY_ACTIVATIONS`: enable activation inspection during play mode.
- `SNAKE_PLAY_ACTIVATION_DIR`: output directory for per-step activation files.
- `SNAKE_PLAY_ACTIVATION_EVERY_STEPS`: export activations every N play steps.
- `SNAKE_PLAY_ACTIVATION_DASHBOARD`: show a separate live pygame dashboard window with labeled color heatmaps.
- `SNAKE_PLAY_STEP_MODE`: pause before each action during activation playback.
- `SNAKE_PLAY_STEP_INTERVAL_SECONDS`: auto-advance interval in step mode, default `2`; set `0` for manual Enter/`r`/`q` controls.

## Activation Dashboard

The activation dashboard is a debugging mode for watching what the trained network sees while Snake is playing. It is intended for qualitative inspection, not training. Swift runs the policy, captures the input stack, `conv1`, `conv2`, dense activations, and Q-values, then streams them to `python/activation_viewer.py` as line-delimited JSON.

Run it with:

```sh
SNAKE_PLAY_ACTIVATIONS=1 \
SNAKE_PLAY_ACTIVATION_DASHBOARD=1 \
SNAKE_PLAY_STEP_MODE=1 \
SNAKE_PLAY_STEP_INTERVAL_SECONDS=2 \
SNAKE_PLAY_ACTIVATION_DIR=runs/model_best_play/activations \
make play-dqn
```

Dashboard behavior:

- The input stack shows four recent binary frames labeled `t-3`, `t-2`, `t-1`, and `t`.
- `conv1` and `conv2` are rendered as tiled channel heatmaps with black padding between tiles.
- Dense activations and Q-values are shown separately so action preferences can be compared against the selected action.
- Spatial heatmaps are transposed for display so they match the pygame gameplay orientation. This is display-only; the model input tensor and saved checkpoint behavior are unchanged.
- `SNAKE_PLAY_STEP_INTERVAL_SECONDS` controls auto-step speed in step mode. The default is `2`; set it to `0` for manual stepping with Enter, `r`, or `q`.

When activation inspection is enabled, play mode also prints the selected action and Q-values for each inspected step. File export writes grayscale PGM heatmaps plus a TSV with per-action Q-values and the selected action.

## Metrics

TensorBoard metrics include:

- `train/loss`
- `train/q_mean_abs`
- `train/q_max_abs`
- `train/grad_l2`
- `train/skipped_update`
- `train/episode_reward`
- `train/episode_score`
- `train/episode_apples`
- `train/episode_steps`
- `train/replay_size`
- `train/replay_fill_ratio`
- `train/epsilon`
- `train/optimize_duration_s`
- `train/action_up`
- `train/action_left`
- `train/action_down`
- `train/action_right`
- `runtime/rss_mb`
- `runtime/vmem_mb`
- `runtime/cpu_user_s`
- `runtime/cpu_system_s`
- `eval/avg_reward`
- `eval/avg_score`
- `eval/avg_apples`

Structured logs include run lifecycle, resume events, episode summaries, optimize-step summaries, evaluation summaries, target syncs, checkpoint saves, GIF saves, stability guards, and resource telemetry.

## Design Decisions

Key decisions made during the project:

- Use Python only for environment ownership because `gym-snake` and pygame rendering already exist there.
- Use Swift for the RL system to keep the DQN implementation, training control, and tests in the Swift package.
- Communicate through typed JSON bridge payloads rather than untyped dictionaries on the Swift side.
- Keep one top-level Swift type per file to make the codebase easier to navigate and review.
- Model actions as `SnakeAction` instead of raw `Int` in DQN data structures.
- Keep observations binary and stacked over 4 frames so the network learns motion and entity roles from temporal evidence.
- Use a `20x20` Snake board resized to `84x84` to keep apple density learnable while preserving the DQN image-input shape.
- Ignore the upstream `gym-snake` 200-step legacy truncation as terminal failure because it artificially caps long successful episodes.
- Sanitize immediate 180-degree snake reversals before sending them to the environment because those invalid actions otherwise dominate early learning.
- Track apples separately from reward/score because shaped or alive rewards make raw score less interpretable.
- Prefer greedy evaluation without epsilon so eval metrics measure policy quality, not exploration noise.
- Use Double DQN to reduce overestimation bias.
- Use PER for long runs because successful and high-TD-error transitions are rare and more informative than uniform random samples.
- Keep PER, Double DQN, epsilon, learning-rate decay, and shaping controlled by environment variables so experiments are reproducible from shell commands.
- Save best checkpoints from evaluation metrics rather than noisy training episode returns.
- Capture best-training-episode GIFs to support qualitative inspection, not as the source of truth for model selection.
- Do not resume optimizer state for now; this keeps checkpointing simple, and model-only resume has been sufficient for these experiments.

## Testing

Python coverage focuses on:

- Adapter reset/step behavior.
- Frame-stack and observation properties.
- Reward configuration validation.
- Python bridge request/response integration.
- Swift-to-Python bridge contract behavior.

Swift coverage focuses on:

- `SnakeEnv` actor behavior.
- Python bridge integration.
- Replay sampling strategies.
- Prioritized replay sampling and priority updates.
- PER beta schedule.
- Epsilon-greedy action selection.
- Learner Bellman math, Double DQN target selection, gradient stability, and stop-gradient behavior.
- Checkpoint manager behavior.
- Environment parser defaults and runtime overrides.
- Learning-rate schedule.
- Structured logger output.
- State tensor layout.

Run the full suite with:

```sh
make test
```

## Toolchain Notes

Swift/MLX command-line runs need an Apple toolchain and MLX Metal support. The Makefile wraps Swift commands with `scripts/with_apple_toolchain.sh` to avoid accidentally picking up Conda or other non-Apple linker binaries.

If you see linker errors, check the linker resolution:

```sh
./scripts/with_apple_toolchain.sh /bin/sh -lc 'which ld; xcrun -f ld'
```

The paths should resolve to Apple/Xcode toolchain locations, not Conda.

DQN targets depend on `prepare-mlx-metallib`, which first runs `build-swift` so a fresh checkout has the MLX Swift dependency checkout available, then calls `scripts/ensure_mlx_metallib.sh` so command-line MLX runs can find `default.metallib`.

## Training History

The meaningful training results are documented in [training.md](training.md). The strongest completed run so far was the 25M-step continuation using GPU, Double DQN, PER, epsilon floor `0.05`, and learning-rate decay from `0.00025` to `0.000125`.

The current best checkpoint is:

```text
checkpoints/model_best.safetensors
```

Use it with:

```sh
make play-dqn
```
