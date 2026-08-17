# A virtualenv puts its interpreter in Scripts/ on Windows and bin/ on everything else.
PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python3)
# Found rather than listed. A wildcard list stops covering a directory the
# moment code moves into a new one, and does it silently: `make check` goes
# on passing while black and pyflakes quietly skip everything that moved.
# The exclusions are the two trees that are not this project's code.
PY_FILES := $(shell find . -name '*.py' \
	-not -path './.venv/*' \
	-not -path './gridfinity-rebuilt-openscad/*' \
	-not -path '*/__pycache__/*')
MD_FILES := $(wildcard *.md) $(wildcard docs/*.md)
DOT_FILES := $(wildcard docs/*.dot)
DOT_SVGS := $(DOT_FILES:.dot=.svg)

# Test workers. Measured on a 16-core box: serial 111s, 8 workers 55s, 16
# workers 58s - past 8 the per-worker cost of importing torch and cv2
# outweighs what another core buys. Risk of OOM at higher counts. Override for a larger machine:
# `make test JOBS=8`.
JOBS ?= 4

.PHONY: format format-check lint typecheck test check check-serial requirements docs docs-check \
	schema schema-check \
	gifs gif-capture gif-pack gif-group gif-drawer previews screenshots solids sheet media

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

# The five checks are independent, so run them at once and let typecheck's
# 13 seconds hide inside the test run. -O groups each target's output
# instead of interleaving pyright's errors into pytest's progress dots.
check:
	@$(MAKE) --no-print-directory -j4 -O format-check lint typecheck test schema-check

# The same checks one at a time, for when interleaved output is the
# problem or a machine cannot spare the cores.
check-serial: format-check lint typecheck schema-check
	$(PYTHON) -m pytest

# Which lockfile this writes depends on where it runs, because
# pip-compile only ever resolves for the machine it is on: a Windows
# run drops the 21 nvidia-* packages and triton, and pins the one
# pyqt5-qt5 that Windows publishes. Writing that over requirements.txt
# would hand Linux a lockfile with no CUDA and the wrong Qt, so the two
# platforms keep their own and neither silently overwrites the other.
REQUIREMENTS_OUT := $(if $(wildcard .venv/Scripts/python.exe),requirements-windows.txt,requirements.txt)

requirements:
	$(PYTHON) -m piptools compile requirements.in --output-file $(REQUIREMENTS_OUT)

# export/schema/*.schema.json, dumped from the Python dict schemas kept
# beside each format's dataclasses - see export/dump_schemas.py.
schema:
	$(PYTHON) -m export.dump_schemas

# In `check`/`check-serial` rather than excluded like docs-check: this
# generation is pure deterministic Python with no external tool whose
# version could shift the output, so staleness here is a real mistake,
# not a machine difference.
schema-check:
	$(PYTHON) -m export.dump_schemas --check

# The rendered SVGs are committed so the docs read on GitHub without a
# graphviz install, which is also why they can go stale - hence docs-check.
docs: $(DOT_SVGS)

%.svg: %.dot
	dot -Tsvg $< -o $@

# Not part of `check`: graphviz is not in requirements.in and a machine
# without it should still be able to run the tests.
#
# Timestamps rather than a re-render and diff: graphviz embeds its own
# version in the output, so comparing bytes would call the SVGs stale on
# any machine with a different graphviz than the one that rendered them.
docs-check:
	@$(MAKE) --no-print-directory -q docs || \
		{ echo "docs/*.svg are older than their .dot sources - run 'make docs'"; exit 1; }

# One reference rendering per drawing path, committed so that a change to
# how anything looks arrives as a diff somebody reads rather than a red
# build somebody silences. See render_demo.py for why these are artifacts
# and not assertions.
#
# A few seconds, so unlike `gifs` this is cheap to re-run on a whim. Still
# not part of `check`: a stale preview is a thing to notice in review, and
# a test that failed on it would be a stored-image assertion by another
# route - the exact thing this is built to avoid.
previews:
	$(PYTHON) -m demos.render_demo --out docs/media

# One picture of each window, for the README. ~2 minutes, nearly all of it
# the floorplan search - see screenshot_demo.py. Needs Qt, driven
# offscreen, so nothing pops open on whoever ran this.
screenshots:
	$(PYTHON) -m demos.screenshot_demo --out docs/media

# The bin itself, rendered by OpenSCAD. ~30 seconds. The only target here
# that needs a tool outside requirements.txt, which is why it is separate
# from `screenshots` rather than folded into it: a machine with Qt but no
# OpenSCAD should still be able to regenerate everything else.
solids:
	$(PYTHON) -m demos.solid_demo --out docs/media

# The calibration sheet, as a picture rather than as the PDF you print.
sheet:
	$(PYTHON) generate_aruco_sheet.py docs/media/sheet_aruco.png

# Everything the README shows. Minutes, and needing Qt, OpenSCAD and a
# cached SAM2 checkpoint between them - so this is a deliberate act rather
# than something `check` does. See docs/media.md.
media: gifs screenshots solids sheet previews

# The README's GIFs. Phony rather than dependency-tracked on purpose: what
# they actually depend on is the behaviour of the whole layout package, and
# a prerequisite list broad enough to be correct would regenerate them - at
# three and a half minutes - every time a test file was touched. They go
# stale silently instead, so regenerate them whenever a change moves parts
# around. `Distribute` has now done that twice.
#
# Three targets rather than one recipe, so a change that only affects
# packing does not cost the 2.5 minutes the grouping search takes. They are
# independent, so `make -j3 gifs` runs them at once - nothing in the layout
# package is threaded, so that is close to a 3x saving on a spare machine.
#
# The same commands appear in docs/media.md, where they document what the
# flags mean; keep the two in step.
gifs: gif-capture gif-pack gif-group gif-drawer

# ~10 seconds, and the only one that records a *window* rather than
# rendering a search - see capture_demo.py.
#
# Two things it needs that the others do not. Qt, which it drives
# offscreen so nothing pops open on whoever ran this. And a SAM2
# checkpoint already in the local Hugging Face cache: the segmenter is
# constructed with local_files_only, so on a machine that has never run
# silhouette_gui.py this target fails rather than quietly downloading a
# few hundred megabytes mid-build.
gif-capture:
	$(PYTHON) -m demos.capture_demo --out docs/media/capture.gif

# ~15 seconds.
gif-pack:
	$(PYTHON) -m demos.layout_demo pack \
		test_data/small_spoon.svg test_data/medium_spoon.svg test_data/big_spoon.svg \
		test_data/medium_fork.svg --out docs/media/pack.gif \
		--restarts 8 --every 8 --pixels-per-mm 1.4 --colors 8

# ~2.5 minutes - the grouping search is quadratic in bins and every
# surviving candidate is a full stochastic solve. See docs/architecture.md.
gif-group:
	$(PYTHON) -m demos.layout_demo group \
		test_data/big_spoon.svg test_data/small_spoon.svg test_data/screwdriver.svg \
		test_data/spreader.svg test_data/big_measure.svg test_data/small_measure.svg \
		--start one-per-bin --restarts 12 --every 1 \
		--out docs/media/group.gif

# ~30 seconds.
gif-drawer:
	$(PYTHON) -m demos.layout_demo drawer \
		test_data/small_spoon.svg test_data/medium_spoon.svg test_data/big_spoon.svg \
		test_data/small_fork.svg test_data/medium_fork.svg test_data/big_fork.svg \
		test_data/spreader.svg test_data/screwdriver.svg \
		test_data/small_measure.svg test_data/big_measure.svg \
		--drawer 210x340 --drawer 170x130 --restarts 6 --every 1 \
		--out docs/media/drawer.gif
