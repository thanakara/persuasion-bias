.DEFAULT_GOAL: help

.PHONY: help lint format all

FILES ?= src/persuasion_bias

help:
	@echo ""
	@echo "--- LINTING ---"
	@echo "lint"
	@echo "lint FILES=<files-to-lint>"
	@echo "format"
	@echo "format FILES=<files-to-format>"
	@echo ""

lint lint_diff:
	@uv run ruff check $(FILES) --diff
	@uv run ruff format $(FILES) --diff

format format_diff:
	@uv run ruff format $(FILES)
	@uv run ruff check --fix $(FILES)

all: lint format