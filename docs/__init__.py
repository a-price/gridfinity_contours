"""Makes `docs` importable, so `rotation_experiment` can run as a module.

That script measures the real packer, so it imports `layout`, which only
resolves with the repository root on `sys.path` - which is what
`python3 -m docs.rotation_experiment` gives it and what
`python3 docs/rotation_experiment.py` does not.

Nothing else here is Python, and nothing imports this package. It exists
so that the one piece of documentation you can *run* is invoked the same
way as everything else in this repository.
"""
