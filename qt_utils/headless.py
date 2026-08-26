r"""Running Qt with nobody at the keyboard, on any platform.

Both demos that photograph a window and the whole test suite need the same
two environment variables set, and both need them set *before* the first Qt
import - Qt reads them when it loads its platform plugin, not later. That
timing is the entire reason this module imports no Qt of its own: anything
that did could not be imported early enough to be useful.

The platform half is the obvious one. The font half is not, and it is
Windows-only. Qt 5.15.2's `offscreen` plugin ships no font backend there,
so it reports zero font families rather than falling back to anything -
and Qt draws no text at all instead of failing. Two things come of that,
neither of which announces itself:

  - Grabbed windows come out with every label, button and title blank.
    `make screenshots` and `make gif-capture` would write exactly that
    into the README, and nothing would return non-zero.
  - Widgets asked how large they want to be answer with metrics computed
    from no font, which ran ~400px wide of the truth here - enough to fail
    a "does this window fit on a screen" assertion for a window that fits
    perfectly well.

`QT_QPA_FONTDIR` points the plugin at the directory the native platform
would have read anyway. Left alone off Windows, where offscreen has
fontconfig and finds fonts without help.
"""

import os
import sys


def UseOffscreenQt() -> None:
    """Ask Qt for a windowless platform that can still draw text.

    `setdefault` throughout, so that someone who has deliberately exported
    either variable - to watch a demo in a real window, or to point at a
    font directory of their own - keeps what they asked for.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
