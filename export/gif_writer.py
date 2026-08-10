"""Writing a sequence of rendered frames out as an animated GIF.

The fourth output device for this project's drawings, after SVG, PDF and
the screen - and the only one whose subject is the *search* rather than
its answer. A printed sheet says where a spoon ended up; a GIF says how
the solver got there, which is the part of this project that is hard to
believe from a still picture.

GIF rather than a video container on purpose: it embeds in a README on
GitHub with no player, no codec, and no external hosting.

Frames arrive as BGR arrays from `layout.render`, which is what OpenCV
produces and what the rest of the pipeline already passes around. The
conversion to RGB happens here, once, rather than at each call site.

**One palette for the whole animation, not one per frame.** This is the
decision the file size turns on, and it is not obvious. A per-frame
adaptive palette is locally optimal and globally terrible: two consecutive
frames that look nearly identical get different palettes, so the same gray
lands on a different index in each, every pixel differs *numerically*, and
the delta encoding that should skip the unchanged 95% of the page finds
nothing to skip. Measured on the drawer animation, per-frame palettes cost
749KB against 196KB for a shared one - and the per-frame version got
*larger* as its palette got smaller, since a coarser adaptive palette
varies more from frame to frame.
"""

from typing import Sequence

import numpy as np
from PIL import Image

# GIF stores frame delays in hundredths of a second, so a duration that is
# not a multiple of 10ms is silently rounded by the decoder - and a frame
# asking for 5ms is usually rounded to 100ms instead, which is why fast
# GIFs written naively come out slower than asked.
DELAY_QUANTUM_MS = 10

# Sixteen levels reproduces an antialiased line drawing to within about 4%
# per channel, which is invisible on an edge and half the size of the 64
# this started at. Raise it for artwork with real color in it.
DEFAULT_COLORS = 16

WHITE = (255, 255, 255)


def Canvas(frames: Sequence[np.ndarray]) -> tuple[int, int]:
    """The smallest `(height, width)` every frame fits inside.

    Frames differ in size whenever the search steps up to a larger bin,
    which is exactly the moment worth animating - so a common canvas is
    required rather than incidental. Public so a caller can report the size
    it is about to write without re-deriving the rule.
    """
    return max(frame.shape[0] for frame in frames), max(frame.shape[1] for frame in frames)


def Padded(frame: np.ndarray, height: int, width: int, fill: tuple[int, int, int] = WHITE) -> np.ndarray:
    """One frame on a larger canvas, anchored at its minimum corner.

    Anchored rather than centered: the drawing's origin is the bin's own
    corner, so pinning it keeps the parts still when the bin around them
    grows. Centering would slide every part sideways at each step up, which
    reads as motion that did not happen.
    """
    if frame.shape[0] > height or frame.shape[1] > width:
        raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, larger than the {width}x{height} canvas")

    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1]] = frame
    return canvas


def Neutral(frames: Sequence[np.ndarray]) -> bool:
    """Whether every pixel of every frame is a shade of gray.

    Detected rather than assumed. Everything this project draws today is
    neutral - the stroke colors in `preview` and `floorplan` are black and
    three grays on white - and that admits a much better palette. But
    baking the assumption in would mean a colored stroke added later came
    out silently desaturated, which is the kind of bug nobody goes looking
    for in an image writer.
    """
    return all((frame[..., 0] == frame[..., 1]).all() and (frame[..., 1] == frame[..., 2]).all() for frame in frames)


def _GrayRamp(levels: int) -> Image.Image:
    """A palette of evenly spaced grays.

    Better than an adaptive palette on this content, and not by a little.
    Antialiasing a gray line against a white page produces exactly a ramp,
    so evenly spaced entries land where the pixels actually are. Median cut
    instead spends its entries where colors *cluster* - crowding near white
    and near black, where the flat regions are - and leaves gaps across the
    middle of the ramp where the edge pixels live. Measured at 16 entries:
    worst-case error 11/255 for the ramp against 37/255 for median cut, at
    the same file size.
    """
    ramp = [round(index * 255 / (levels - 1)) for index in range(levels)]
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for value in ramp for channel in (value, value, value)])
    return palette


def _Composite(frames: Sequence[np.ndarray], colors: int) -> Image.Image:
    """A palette covering every frame, for content this cannot assume is
    gray.

    Derived from all the frames stacked together rather than from one of
    them: the first frame of a search is nearly empty and the last is full,
    so a palette taken from either would misrepresent the rest.
    """
    stacked = np.concatenate([frame[..., ::-1] for frame in frames], axis=0)
    return Image.fromarray(stacked).quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def _Palette(frames: Sequence[np.ndarray], colors: int) -> Image.Image:
    return _GrayRamp(colors) if Neutral(frames) else _Composite(frames, colors)


def WriteGif(
    path: str,
    frames: Sequence[np.ndarray],
    milliseconds_per_frame: int = 60,
    colors: int = DEFAULT_COLORS,
    fill: tuple[int, int, int] = WHITE,
) -> None:
    """Write BGR frames as a looping animated GIF.

    Frames may differ in size; they are padded to a common canvas. Every
    frame gets the same delay - a variable one would need the caller to
    decide what "the algorithm slowing down" should mean, and it does not
    mean anything here.

    Note that the file will usually hold fewer frames than were passed:
    `optimize` collapses a run of identical frames into one shown for their
    combined time. That is why holding on a final result costs a handful of
    bytes rather than a copy of the image per frame - repeat the frame and
    let the encoder do it.

    `colors` sizes the one palette every frame shares. Dithering is off
    throughout: it would trade a flat white page for a stippled one, which
    costs both file size and legibility on a line drawing.
    """
    if not frames:
        raise ValueError("an animation needs at least one frame")
    for index, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"frame {index} is not an 8-bit BGR image: shape {frame.shape}, dtype {frame.dtype}")
    if milliseconds_per_frame < DELAY_QUANTUM_MS:
        raise ValueError(f"a GIF frame cannot be shown for less than {DELAY_QUANTUM_MS}ms")
    if not 2 <= colors <= 256:
        raise ValueError(f"a GIF palette holds 2 to 256 colors, got {colors}")

    height, width = Canvas(frames)
    padded = [Padded(frame, height, width, fill) for frame in frames]

    palette = _Palette(padded, colors)
    images = [Image.fromarray(frame[..., ::-1]).quantize(palette=palette, dither=Image.Dither.NONE) for frame in padded]

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=DELAY_QUANTUM_MS * round(milliseconds_per_frame / DELAY_QUANTUM_MS),
        loop=0,  # forever; 1 would mean "play twice", which is nobody's intent
        optimize=True,
        disposal=1,  # leave each frame up, so a partial redraw does not flash white
    )
