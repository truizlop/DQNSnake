SHELL := /bin/zsh

PYTHON ?= python3
SWIFT_DIR := swift
SWIFT_ENV := ./scripts/with_apple_toolchain.sh

.DEFAULT_GOAL := help

.PHONY: help setup build build-swift test test-python test-swift run run-smoke run-visual run-swift run-swift-visual run-dqn play-dqn prepare-mlx-metallib open-xcode clean

help: ## Show available targets
	@echo "Snake DQN - Main Commands"
	@echo ""
	@grep -E '^[a-zA-Z0-9._-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup: ## Install Python test/runtime dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install gym gym-snake numpy pytest

build: build-swift ## Build all supported components

build-swift: ## Build Swift package
	@$(SWIFT_ENV) swift build --package-path $(SWIFT_DIR)

test: test-python test-swift ## Run all tests

test-python: ## Run Python tests
	pytest -q

test-swift: ## Run Swift tests
	@$(SWIFT_ENV) swift test --package-path $(SWIFT_DIR)

run: run-smoke ## Run main smoke run

run-smoke: ## Run Python snake env smoke script
	PYTHONPATH=python $(PYTHON) tests/smoke_snake_env.py

run-visual: ## Run visual snake loop (SNAKE_CONTROL=human|random)
	PYTHONPATH=python $(PYTHON) tests/visual_snake_env.py

run-swift: ## Run Swift CLI that controls snake via Python bridge
	@SNAKE_PYTHON_DIR=python \
	SNAKE_PYTHON_EXE=/opt/anaconda3/bin/python3 \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-env-cli

run-swift-visual: ## Run Swift-controlled snake with pygame rendering
	@SNAKE_PYTHON_DIR=python \
	SNAKE_PYTHON_EXE=/opt/anaconda3/bin/python3 \
	SNAKE_VISUAL=1 \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-env-cli

prepare-mlx-metallib: ## Build MLX metallib required by command-line MLX runs
	@./scripts/ensure_mlx_metallib.sh

run-dqn: ## Run DQN trainer
	@$(SWIFT_ENV) ./scripts/ensure_mlx_metallib.sh; \
	SNAKE_PYTHON_DIR=python \
	SNAKE_PYTHON_EXE=/opt/anaconda3/bin/python3 \
	SNAKE_MLX_DEVICE=$${SNAKE_MLX_DEVICE:-cpu} \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-dqn-train

play-dqn: ## Play Snake using the best trained DQN checkpoint
	@$(SWIFT_ENV) ./scripts/ensure_mlx_metallib.sh; \
	SNAKE_PYTHON_DIR=python \
	SNAKE_PYTHON_EXE=/opt/anaconda3/bin/python3 \
	SNAKE_MLX_DEVICE=$${SNAKE_MLX_DEVICE:-gpu} \
	SNAKE_PLAY_ONLY=1 \
	$(SWIFT_ENV) swift run --package-path $(SWIFT_DIR) snake-dqn-train

open-xcode: ## Open Swift package in Xcode with Apple toolchain-sanitized environment
	@./scripts/open_xcode_with_toolchain.sh

clean: ## Remove build/test caches
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf $(SWIFT_DIR)/.build
