# parsimony — developer task runner. Same commands locally and in CI.
.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/python -m pip
export PYTHONPATH := src

.PHONY: help setup fmt lint typecheck security test cov-critical audit check docs serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dev deps + pre-commit hooks
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,graph]"
	-$(VENV)/bin/pre-commit install

fmt: ## Auto-format
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

lint: ## Lint (no changes)
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

typecheck: ## Strict type check
	MYPYPATH=src $(VENV)/bin/mypy src

security: ## SAST (bandit)
	$(VENV)/bin/bandit -q -r src

audit: ## Dependency vulnerability scan (SCA)
	$(VENV)/bin/pip-audit || echo "pip-audit: review findings"

test: ## Run tests with the >=90% line+branch coverage gate
	$(PY) -m pytest

cov-critical: ## Enforce 100% coverage on critical paths (transform/decision/detection core)
	$(PY) -m pytest -o addopts="" \
	  --cov=parsimony.compress --cov=parsimony.model --cov=parsimony.spans \
	  --cov=parsimony.cache.key --cov=parsimony.cache.policy --cov=parsimony.redact \
	  --cov=parsimony.invariants --cov=parsimony.eval.equivalence --cov=parsimony.eval.quality \
	  --cov=parsimony.memory.compaction --cov=parsimony.tenant \
	  --cov-branch --cov-fail-under=100 --cov-report=term-missing

eval: ## Measure token savings + lossless equivalence over the built-in corpus
	$(PY) -m parsimony.cli eval

docs: ## Generate API docs (pdoc)
	$(VENV)/bin/pdoc -o site parsimony

check: lint typecheck security test cov-critical ## Everything CI runs

serve: ## Run the proxy locally (loopback + tailnet)
	$(PY) -m parsimony.cli serve

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage site build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
