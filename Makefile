PYTHON := .venv/bin/python3
PY_FILES := $(wildcard *.py) $(wildcard pipeline/*.py)
MD_FILES := $(wildcard *.md)

.PHONY: format format-check lint typecheck test check

format:
	$(PYTHON) -m black $(PY_FILES)
	$(PYTHON) -m mdformat $(MD_FILES)

format-check:
	$(PYTHON) -m black --check $(PY_FILES)
	$(PYTHON) -m mdformat --check $(MD_FILES)

lint:
	$(PYTHON) -m pyflakes $(PY_FILES)

typecheck:
	$(PYTHON) -m pyright

test:
	$(PYTHON) -m pytest

check: format-check lint typecheck test
