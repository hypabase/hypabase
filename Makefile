.PHONY: lint format format-check typecheck test check build clean docs-serve docs-build docs-deploy docs-lint

lint:
	uv run ruff check hypabase/ tests/

format:
	uv run ruff format hypabase/ tests/

format-check:
	uv run ruff format --check hypabase/ tests/

typecheck:
	uv run mypy hypabase/

test:
	uv run pytest

check: lint format-check typecheck test
	@echo "All checks passed."

build: check
	uv build

clean:
	rm -rf dist/ build/ *.egg-info .mypy_cache .ruff_cache .pytest_cache site/

docs-serve:
	uv run --extra docs mkdocs serve

docs-build:
	uv run --extra docs mkdocs build --strict

docs-deploy:
	uv run --extra docs mkdocs gh-deploy --force

docs-lint:
	vale docs/
