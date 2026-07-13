PYTHON := .venv/bin/python3
PY_FILES := $(wildcard *.py)

.PHONY: format format-check lint typecheck test check

format:
	$(PYTHON) -m black $(PY_FILES)

format-check:
	$(PYTHON) -m black --check $(PY_FILES)

lint:
	$(PYTHON) -m pyflakes $(PY_FILES)

typecheck:
	$(PYTHON) -m pyright

test:
	$(PYTHON) -m pytest

check: format-check lint typecheck test
