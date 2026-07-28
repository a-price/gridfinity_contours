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
"""

from typing import Sequence

import numpy as np
from PIL import Image

# GIF stores frame delays in hundredths of a second, so a duration that is
# not a multiple of 10ms is silently rounded by the decoder - and a frame
# asking for 5ms is usually rounded to 100ms instead, which is why fast
# GIFs written naively come out slower than asked.
DELAY_QUANTUM_MS = 10

# The drawings are dark lines on white with antialiasing, so almost all of
# the color range is grays that nobody can distinguish. Quantizing hard is
# what keeps a few hundred frames to a few hundred kilobytes.
DEFAULT_COLORS = 64

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


def _Quantized(frame: np.ndarray, colors: int) -> Image.Image:
    """One BGR frame as a paletted image, without dithering.

    Dithering trades a flat white page for a stippled one, which costs both
    file size and legibility here - the frames are line drawings, so a
    palette of grays reproduces them essentially exactly.
    """
    rgb = Image.fromarray(frame[..., ::-1])
    return rgb.convert("P", dither=Image.Dither.NONE, palette=Image.Palette.ADAPTIVE, colors=colors)


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
    """
    if not frames:
        raise ValueError("an animation needs at least one frame")
    for index, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"frame {index} is not an 8-bit BGR image: shape {frame.shape}, dtype {frame.dtype}")
    if milliseconds_per_frame < DELAY_QUANTUM_MS:
        raise ValueError(f"a GIF frame cannot be shown for less than {DELAY_QUANTUM_MS}ms")

    height, width = Canvas(frames)
    images = [_Quantized(Padded(frame, height, width, fill), colors) for frame in frames]

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=DELAY_QUANTUM_MS * round(milliseconds_per_frame / DELAY_QUANTUM_MS),
        loop=0,  # forever; 1 would mean "play twice", which is nobody's intent
        optimize=True,
        disposal=1,  # leave each frame up, so a partial redraw does not flash white
    )
