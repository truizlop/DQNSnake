SHELL := /bin/zsh

# DQNSnake command reference
# ==========================
#
# This Makefile is the operational entry point for the current implementation:
#
# - Python owns the gym-snake environment adapter and JSON bridge.
# - Swift owns the DQN implementation, replay buffer, training scheduler,
#   evaluation loop, checkpointing, TensorBoard/structured metrics, and play mode.
# - MLX command-line runs need a generated `default.metallib`; DQN targets build
#   Swift dependencies and call `prepare-mlx-metallib` automatically.
#
# Common commands:
#
#   make setup
#   make test
#   make play-dqn
#   SNAKE_RESUME_CHECKPOINT=checkpoints/model_best.safetensors make eval-dqn
#   SNAKE_MLX_DEVICE=gpu SNAKE_STEPS=25000000 make train-dqn
#
# Important runtime defaults:
#
# - `play-dqn` defaults to `checkpoints/model_best.safetensors`, GPU, one rendered
#   greedy episode, and GIF output under `runs/model_best_play`.
# - `eval-dqn` is greedy evaluation only. It defaults to `model_best` and disables
#   TensorBoard/browser launch, structured logs, and training GIF capture.
# - `train-dqn` uses the Swift baseline in `DQNHyperparameterBaseline.snakeV1`.
#   Override it with `SNAKE_*` environment variables.
#
# Core DQN environment variables:
#
# - `SNAKE_STEPS`: total environment steps for training.
# - `SNAKE_RESUME_CHECKPOINT`: `.safetensors` checkpoint path.
# - `SNAKE_MLX_DEVICE`: `cpu` or `gpu`.
# - `SNAKE_DQN_ALGORITHM`: `single` or `double`.
# - `SNAKE_REPLAY_SAMPLING_STRATEGY`: `with_replacement`, `without_replacement`, or `prioritized`.
# - `SNAKE_PER_ALPHA`, `SNAKE_PER_BETA_START`, `SNAKE_PER_BETA_ANNEAL_STEPS`, `SNAKE_PER_EPSILON`: PER controls.
# - `SNAKE_EPSILON_START`, `SNAKE_EPSILON_END`, `SNAKE_EPSILON_DECAY_STEPS`: exploration schedule.
# - `SNAKE_LEARNING_RATE`, `SNAKE_LEARNING_RATE_FINAL`,
#   `SNAKE_LEARNING_RATE_DECAY_START_STEP`, `SNAKE_LEARNING_RATE_DECAY_END_STEP`: learning-rate schedule.
# - `SNAKE_EVAL_EPISODES`, `SNAKE_EVAL_EVERY_EPISODES`, `SNAKE_EVAL_SEEDS`: evaluation controls.
# - `SNAKE_TB_ENABLE`, `SNAKE_TB_LAUNCH`, `SNAKE_TB_LOGDIR`, `SNAKE_TB_PORT`: TensorBoard controls.
# - `SNAKE_OBS_ENABLE`, `SNAKE_OBS_LOG_PATH`: structured observability log controls.
# - `SNAKE_BEST_GIF_ENABLE`, `SNAKE_BEST_GIF_DIR`: training best-episode GIF capture.
# - `SNAKE_PLAY_EPISODES`, `SNAKE_PLAY_RENDER`, `SNAKE_PLAY_GIF_DIR`: play-mode controls.
# - `SNAKE_PLAY_ACTIVATIONS`, `SNAKE_PLAY_ACTIVATION_DIR`,
#   `SNAKE_PLAY_ACTIVATION_EVERY_STEPS`, `SNAKE_PLAY_ACTIVATION_DASHBOARD`,
#   `SNAKE_PLAY_STEP_MODE`, `SNAKE_PLAY_STEP_INTERVAL_SECONDS`: activation debug controls.
#
# Useful long-run example:
#
#   SNAKE_MLX_DEVICE=gpu \
#   SNAKE_STEPS=25000000 \
#   SNAKE_RESUME_CHECKPOINT=checkpoints/model_step_14990000.safetensors \
#   SNAKE_DQN_ALGORITHM=double \
#   SNAKE_REPLAY_SAMPLING_STRATEGY=prioritized \
#   SNAKE_PER_ALPHA=0.6 \
#   SNAKE_PER_BETA_START=0.4 \
#   SNAKE_PER_BETA_ANNEAL_STEPS=1000000 \
#   SNAKE_PER_EPSILON=0.001 \
#   SNAKE_EPSILON_END=0.05 \
#   SNAKE_LEARNING_RATE=0.00025 \
#   SNAKE_LEARNING_RATE_FINAL=0.000125 \
#   SNAKE_LEARNING_RATE_DECAY_START_STEP=15000000 \
#   SNAKE_LEARNING_RATE_DECAY_END_STEP=25000000 \
#   make train-dqn

PYTHON ?= python3
SWIFT_DIR := swift
SWIFT_ENV := ./scripts/with_apple_toolchain.sh
PYTHON_EXE ?= /opt/anaconda3/bin/python3
PYTHON_DIR ?= python

.DEFAULT_GOAL := help

.PHONY: \
	help docs \
	setup build build-swift \
	test test-python test-swift \
	smoke-python visual-python smoke-swift visual-swift \
	train-dqn run-dqn eval-dqn play-dqn \
	prepare-mlx-metallib open-xcode \
	clean clean-build-artifacts \
	run run-smoke run-visual run-swift run-swift-visual

help: ## Show concise command list
	@echo "Snake DQN - current commands"
	@echo ""
	@grep -E '^[a-zA-Z0-9._-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  %-24s %s\n", $$1, $$2}'
	@echo ""
	@echo "Run 'make docs' for workflow notes and important SNAKE_* overrides."

docs: ## Print the Makefile documentation header
	@sed -n '1,64p' $(MAKEFILE_LIST)

setup: ## Install Python runtime/test dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install gym gym-snake numpy pytest pillow pygame tensorboard

build: build-swift ## Build all compiled components

build-swift: ## Build the Swift package
	@$(SWIFT_ENV) swift build --package-path $(SWIFT_DIR)

test: test-python test-swift ## Run Python and Swift tests

test-python: ## Run Python bridge/adapter tests
	PYTHONPATH=$(PYTHON_DIR) pytest -q

test-swift: ## Run Swift package tests
	@$(SWIFT_ENV) swift test --package-path $(SWIFT_DIR)

smoke-python: ## Run Python snake environment smoke test
	PYTHONPATH=$(PYTHON_DIR) $(PYTHON) tests/smoke_snake_env.py

visual-python: ## Run Python visual snake loop; set SNAKE_CONTROL=human for keyboard control
	PYTHONPATH=$(PYTHON_DIR) $(PYTHON) tests/visual_snake_env.py

smoke-swift: ## Run Swift bridge CLI with random actions, no rendering
	@SNAKE_PYTHON_DIR=$(PYTHON_DIR) \
	SNAKE_PYTHON_EXE=$(PYTHON_EXE) \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-env-cli

visual-swift: ## Run Swift bridge CLI with random actions and pygame rendering
	@SNAKE_PYTHON_DIR=$(PYTHON_DIR) \
	SNAKE_PYTHON_EXE=$(PYTHON_EXE) \
	SNAKE_VISUAL=1 \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-env-cli

prepare-mlx-metallib: build-swift ## Build/copy MLX metallib needed by command-line MLX runs
	@$(SWIFT_ENV) ./scripts/ensure_mlx_metallib.sh

train-dqn: prepare-mlx-metallib ## Run Swift DQN training; configure with SNAKE_* environment variables
	@SNAKE_PYTHON_DIR=$(PYTHON_DIR) \
	SNAKE_PYTHON_EXE=$(PYTHON_EXE) \
	SNAKE_MLX_DEVICE=$${SNAKE_MLX_DEVICE:-cpu} \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-dqn-train

run-dqn: train-dqn ## Backward-compatible alias for train-dqn

eval-dqn: prepare-mlx-metallib ## Greedily evaluate a checkpoint; defaults to checkpoints/model_best.safetensors
	@SNAKE_PYTHON_DIR=$(PYTHON_DIR) \
	SNAKE_PYTHON_EXE=$(PYTHON_EXE) \
	SNAKE_MLX_DEVICE=$${SNAKE_MLX_DEVICE:-gpu} \
	SNAKE_EVAL_ONLY=1 \
	SNAKE_RESUME_CHECKPOINT=$${SNAKE_RESUME_CHECKPOINT:-checkpoints/model_best.safetensors} \
	SNAKE_TB_ENABLE=0 \
	SNAKE_TB_LAUNCH=0 \
	SNAKE_OBS_ENABLE=0 \
	SNAKE_BEST_GIF_ENABLE=0 \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-dqn-train

play-dqn: prepare-mlx-metallib ## Play Snake with the best trained model; renders by default and exports a GIF
	@SNAKE_PYTHON_DIR=$(PYTHON_DIR) \
	SNAKE_PYTHON_EXE=$(PYTHON_EXE) \
	SNAKE_MLX_DEVICE=$${SNAKE_MLX_DEVICE:-gpu} \
	SNAKE_PLAY_ONLY=1 \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-dqn-train

open-xcode: ## Open Swift package in Xcode with Apple toolchain-sanitized environment
	@./scripts/open_xcode_with_toolchain.sh

clean: clean-build-artifacts ## Remove build/test caches only

clean-build-artifacts: ## Remove local build caches and Python bytecode; keep runs/checkpoints
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf $(SWIFT_DIR)/.build

# Backward-compatible aliases for older README/conversation commands.
run: smoke-python ## Alias for smoke-python
run-smoke: smoke-python ## Alias for smoke-python
run-visual: visual-python ## Alias for visual-python
run-swift: smoke-swift ## Alias for smoke-swift
run-swift-visual: visual-swift ## Alias for visual-swift
