r"""Make a Windows virtualenv able to import both Qt and torch.

**The conflict.** PyQt5's Windows wheel is `pyqt5-qt5`, and 5.15.2 is the
only version that has ever been published for Windows - there is no newer
one to upgrade to. It bundles its own copy of the MSVC runtime
(`msvcp140.dll` and friends) built in 2020, version 14.26. Importing PyQt5
puts that directory on the DLL search path, so those are the copies the
process gets. torch's `c10.dll` is built against a much later runtime, and
when it loads into a process that already holds 14.26 it fails
initialization outright:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading ...\torch\lib\c10.dll

Only in that order. `import torch` before `import PyQt5` is fine, which is
what makes this so confusing to meet: three of the four GUIs never import
torch and are unaffected, and `silhouette_gui.py` imports Qt at line 29 and
the segmenter at line 45, so it fails while a bare `import torch` works.

**Why this is not a pin.** It is tempting to read it as version drift and
reach for requirements.txt. It is not: the pinned torch 2.12.1 fails
exactly the same way, and there is no `pyqt5-qt5` other than 5.15.2 to
pin to. Nor is it something a clean install fixes - the stale DLLs come
out of the wheel, so a fresh venv reproduces it precisely.

**The fix.** Windows itself ships a current MSVC runtime (14.51 at the
time of writing) in System32, and it is backward compatible with what Qt
5.15.2 needs. Renaming the bundled copies aside lets both libraries load
the system one, and Qt is no worse off for it. Renaming rather than
deleting so the change is reversible and visible.

This is not something the venv can remember, so re-run it after any
`pip install` that reinstalls PyQt5. It is safe to run repeatedly, and it
does nothing at all off Windows.
"""

import os
import subprocess
import sys

# The runtime that Qt bundles and that torch cannot live with. `concrt140`
# is the concurrency runtime and the `msvcp140_*` files are the C++
# standard library split across several DLLs - all part of the same
# redistributable, so they move together or not at all.
BUNDLED_RUNTIME_DLLS = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
)

SUFFIX = ".bak"


def QtBinDirectory() -> str:
    """Where PyQt5 keeps the Qt DLLs, found through PyQt5 itself.

    Imported here rather than at module scope so that the script can say
    something useful on a machine where PyQt5 is not installed yet,
    instead of dying on its own import line.
    """
    import PyQt5

    return os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "bin")


def RetireBundledRuntime(qt_bin: str) -> list[str]:
    """Rename the bundled runtime aside. Returns what was moved.

    An empty list means there was nothing left to do, which is the normal
    result of a second run rather than a failure.
    """
    moved = []
    for name in BUNDLED_RUNTIME_DLLS:
        source = os.path.join(qt_bin, name)
        if not os.path.exists(source):
            continue
        os.replace(source, os.path.join(qt_bin, name + SUFFIX))
        moved.append(name)
    return moved


def ImportsSurviveInBothOrders() -> bool:
    """Whether Qt and torch now load regardless of which goes first.

    In a subprocess because the point is what a *fresh* interpreter does:
    this one has already imported whatever it has imported, and a DLL that
    is loaded stays loaded.
    """
    for first, second in (("PyQt5.QtWidgets", "torch"), ("torch", "PyQt5.QtWidgets")):
        finished = subprocess.run(
            [sys.executable, "-c", f"import {first}; import {second}"],
            capture_output=True,
        )
        if finished.returncode != 0:
            return False
    return True


def main() -> int:
    if sys.platform != "win32":
        print(f"Not Windows ({sys.platform}) - nothing to do.")
        return 0

    try:
        qt_bin = QtBinDirectory()
    except ImportError:
        print("PyQt5 is not installed in this interpreter - install requirements first.")
        return 1

    if not os.path.isdir(qt_bin):
        print(f"No Qt DLL directory at {qt_bin} - is this really PyQt5's Windows wheel?")
        return 1

    moved = RetireBundledRuntime(qt_bin)
    if moved:
        print(f"Retired {len(moved)} bundled runtime DLL(s) in {qt_bin}:")
        for name in moved:
            print(f"  {name} -> {name}{SUFFIX}")
    else:
        print(f"Nothing to retire in {qt_bin} - already done.")

    # Checked rather than assumed: the whole point of the script is that
    # this import pair works, and saying so without testing it is how a
    # setup step starts lying about an environment it no longer fixes.
    if ImportsSurviveInBothOrders():
        print("Verified: Qt and torch import in either order.")
        return 0

    print("Qt and torch still conflict - the rename was not enough. See this file's docstring.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
