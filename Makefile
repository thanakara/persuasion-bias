.DEFAULT_GOAL: help

.PHONY: help lint format test all

FILES ?= src/persuasion_bias

help:
	@echo ""
	@echo "--- LINTING ---"
	@echo "lint"
	@echo "lint FILES=..."
	@echo "format"
	@echo "format FILES=..."
	@echo "--- TESTING ---"
	@echo "test"
	@echo ""

lint lint_diff:
	@uv run ruff check $(FILES) --diff
	@uv run ruff format $(FILES) --diff

format format_diff:
	@uv run ruff format $(FILES)
	@uv run ruff check --fix $(FILES)

test:
	@uv run pytest tests/ --no-header -v

all: lint format test