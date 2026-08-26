"""Qt helpers with no knowledge of this project in them.

    widgets          labeled slider, spin box, combo box, group box
    click_recorder   clicks on a displayed image, in the image's own pixels
    window_capture   a window as an array, once it has stopped changing
    headless         the environment Qt needs to run with no display

Nothing here mentions a contour, a bin or a drawer. That is the whole
membership test, and it is what separates this from `capture.pipeline`: a
`Stage` and a `Pipeline` are this project's idea of how a capture step is
structured, while a spin box that waits for you to stop typing is not.

Two of these exist because Qt's obvious signal is the wrong one. A control
edited continuously must not report every intermediate value, or whatever
it drives re-runs on each one; `widgets` holds that policy in one place
rather than in each panel. And a window grabbed the moment it is asked has
not necessarily finished laying itself out, so `window_capture` grabs
until two frames agree - which is why the documentation screenshots are
not half-drawn.

`click_recorder` is the one with real arithmetic: a QLabel scales and
letterboxes the pixmap it shows, so a click's widget coordinates are not
the image's. Getting that mapping wrong puts the segmenter's seed points
somewhere the user did not click, which looks like a bad model rather than
a bad transform.
"""
