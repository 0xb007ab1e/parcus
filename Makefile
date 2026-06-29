# parcus — developer task runner. Same commands locally and in CI.
.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/python -m pip
# Mutation tester is NOT a project dependency — run it ephemerally. Override to use a local
# install, e.g. `make mutate MUTMUT=$(VENV)/bin/mutmut`.
MUTMUT ?= uvx mutmut
export PYTHONPATH := src

.PHONY: help setup fmt lint typecheck security test cov-critical mutate audit check docs serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dev deps + pre-commit hooks
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,graph]"
	-$(VENV)/bin/pre-commit install

fmt: ## Auto-format
	$(VENV)/bin/ruff format src tests scripts
	$(VENV)/bin/ruff check --fix src tests scripts

lint: ## Lint (no changes)
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

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
	  --cov=parcus.compress --cov=parcus.model --cov=parcus.spans \
	  --cov=parcus.cache.key --cov=parcus.cache.policy --cov=parcus.redact \
	  --cov=parcus.invariants --cov=parcus.eval.equivalence --cov=parcus.eval.quality \
	  --cov=parcus.memory.compaction --cov=parcus.memory.provider --cov=parcus.tenant \
	  --cov=parcus.quota --cov=parcus.cache.similarity --cov=parcus.cache.encryption \
	  --cov-branch --cov-fail-under=100 --cov-report=term-missing

mutate: ## Mutation-test the critical modules (ephemeral via uvx; slow — nightly/on-demand)
	$(MUTMUT) run
	$(MUTMUT) results

eval: ## Measure token savings + lossless equivalence over the built-in corpus
	$(PY) -m parcus.cli eval

docs: ## Generate API docs (pdoc)
	$(VENV)/bin/pdoc -o site parcus

docs-links: ## Check Markdown for broken relative links + anchors (hermetic)
	$(PY) scripts/check_links.py

check: lint typecheck security test cov-critical docs-links ## Everything CI runs

serve: ## Run the proxy locally (loopback + tailnet)
	$(PY) -m parcus.cli serve

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage site build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
