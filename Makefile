PYTHON := .venv/bin/python3
PY_FILES := $(wildcard *.py) $(wildcard pipeline/*.py) $(wildcard pipeline/*/*.py)
MD_FILES := $(wildcard *.md) $(wildcard docs/*.md)
DOT_FILES := $(wildcard docs/*.dot)
DOT_SVGS := $(DOT_FILES:.dot=.svg)

# Test workers. Measured on a 16-core box: serial 111s, 8 workers 55s, 16
# workers 58s - past 8 the per-worker cost of importing torch and cv2
# outweighs what another core buys. Override for a smaller machine:
# `make test JOBS=4`.
JOBS ?= 8

.PHONY: format format-check lint typecheck test check check-serial requirements docs docs-check

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

# Parallel here rather than in pyproject's addopts, so that running pytest
# directly on one test stays serial - live output and a readable traceback
# matter more than speed when the thing has already failed.
test:
	$(PYTHON) -m pytest -n $(JOBS)

# The four checks are independent, so run them at once and let typecheck's
# 13 seconds hide inside the test run. -O groups each target's output
# instead of interleaving pyright's errors into pytest's progress dots.
check:
	@$(MAKE) --no-print-directory -j4 -O format-check lint typecheck test

# The same checks one at a time, for when interleaved output is the
# problem or a machine cannot spare the cores.
check-serial: format-check lint typecheck
	$(PYTHON) -m pytest

requirements:
	$(PYTHON) -m piptools compile requirements.in --output-file requirements.txt

# The rendered SVGs are committed so the docs read on GitHub without a
# graphviz install, which is also why they can go stale - hence docs-check.
docs: $(DOT_SVGS)

%.svg: %.dot
	dot -Tsvg $< -o $@

# Not part of `check`: graphviz is not in requirements.in and a machine
# without it should still be able to run the tests.
docs-check: docs
	@git diff --quiet -- $(DOT_SVGS) || \
		{ echo "docs/*.svg are stale - commit the re-rendered files"; exit 1; }
