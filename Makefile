SHELL := /bin/zsh

PYTHON ?= python3
SWIFT_DIR := swift

.DEFAULT_GOAL := help

.PHONY: help setup build build-swift test test-python test-swift run run-smoke run-visual run-swift run-swift-visual run-dqn clean

help: ## Show available targets
	@echo "Snake DQN - Main Commands"
	@echo ""
	@grep -E '^[a-zA-Z0-9._-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup: ## Install Python test/runtime dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install gym gym-snake numpy pytest

build: build-swift ## Build all supported components

build-swift: ## Build Swift package
	@PATH="$${PATH//:\/opt\/anaconda3\/bin/}"; PATH="$${PATH//\/opt\/anaconda3\/bin:/}"; \
	swift build --package-path $(SWIFT_DIR)

test: test-python test-swift ## Run all tests

test-python: ## Run Python tests
	pytest -q

test-swift: ## Run Swift tests
	@PATH="$${PATH//:\/opt\/anaconda3\/bin/}"; PATH="$${PATH//\/opt\/anaconda3\/bin:/}"; \
	swift test --package-path $(SWIFT_DIR)

run: run-smoke ## Run main smoke run

run-smoke: ## Run Python snake env smoke script
	PYTHONPATH=python $(PYTHON) tests/smoke_snake_env.py

run-visual: ## Run visual snake loop (SNAKE_CONTROL=human|random)
	PYTHONPATH=python $(PYTHON) tests/visual_snake_env.py

run-swift: ## Run Swift CLI that controls snake via Python bridge
	@PATH="$${PATH//:\/opt\/anaconda3\/bin/}"; PATH="$${PATH//\/opt\/anaconda3\/bin:/}"; \
	SNAKE_PYTHON_DIR=python \
	SNAKE_PYTHON_EXE=/opt/anaconda3/bin/python3 \
	swift run --package-path $(SWIFT_DIR) snake-env-cli

run-swift-visual: ## Run Swift-controlled snake with pygame rendering
	@PATH="$${PATH//:\/opt\/anaconda3\/bin/}"; PATH="$${PATH//\/opt\/anaconda3\/bin:/}"; \
	SNAKE_PYTHON_DIR=python \
	SNAKE_PYTHON_EXE=/opt/anaconda3/bin/python3 \
	SNAKE_VISUAL=1 \
	swift run --package-path $(SWIFT_DIR) snake-env-cli

run-dqn: ## Run DQN scaffold (set SNAKE_ENABLE_MLX=1 for MLX tensor ops)
	@PATH="$${PATH//:\/opt\/anaconda3\/bin/}"; PATH="$${PATH//\/opt\/anaconda3\/bin:/}"; \
	SNAKE_PYTHON_DIR=python \
	SNAKE_PYTHON_EXE=/opt/anaconda3/bin/python3 \
	swift run --package-path $(SWIFT_DIR) snake-dqn-train

clean: ## Remove build/test caches
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf $(SWIFT_DIR)/.build
