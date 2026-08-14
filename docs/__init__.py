"""Makes `docs` importable, so `rotation_experiment` can run as a module.

That script measures the real packer, so it imports `layout`, which only
resolves with the repository root on `sys.path` - which is what
`python3 -m docs.rotation_experiment` gives it and what
`python3 docs/rotation_experiment.py` does not.

`solid.py` sits here too and needs none of this: it imports nothing from
this project and runs as a plain script. Nothing else in `docs/` is
Python at all. This package exists so that the one piece of documentation
you can run *against the packer* is invoked the same way as everything
else in this repository.
"""
