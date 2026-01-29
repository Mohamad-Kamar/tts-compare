# TTS Compare - Makefile

.PHONY: help setup-kokoro setup-chatterbox setup-qwen3 test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup (creates separate venv per engine)
setup-kokoro:  ## Setup Kokoro (.venv-kokoro)
	@./scripts/setup.sh kokoro

setup-chatterbox:  ## Setup Chatterbox (.venv-chatterbox)
	@./scripts/setup.sh chatterbox

setup-qwen3:  ## Setup Qwen3 (.venv-qwen3)
	@./scripts/setup.sh qwen3

# Testing & Linting (run inside activated venv)
test:  ## Run tests
	pytest tests/ -v

test-fast:  ## Run tests (skip slow)
	pytest tests/ -v -m "not slow"

lint:  ## Run linter
	ruff check engines/ cli/ utils/ tests/

lint-fix:  ## Fix lint issues
	ruff check engines/ cli/ utils/ tests/ --fix

# Cleanup
clean:  ## Remove all venvs and caches
	rm -rf .venv-*/ __pycache__/ .pytest_cache/ .ruff_cache/ *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Quick demos
demo:  ## Quick demo (requires activated venv)
	tts generate -e kokoro -t "Hello, this is TTS Compare." -o output/demo.wav

engines:  ## List installed engines
	tts engines
