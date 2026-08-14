import numpy as np

from capture.morphology import Morphology


def _make_notched_mask() -> np.ndarray:
    """A rectangle (cols 40-59, rows 10-129) with a notch cut out of its
    upper-left interior. The notch is asymmetric mass, but - unlike an
    added bump - it doesn't change the rectangle's outer extent, so the
    bounding box stays centered at x=49.5 regardless of the notch.
    """
    mask = np.zeros((140, 100), dtype=bool)
    mask[10:130, 40:60] = True
    mask[20:30, 40:45] = False  # notch: left side, interior only
    return mask


def _morphology(**symmetry_kwargs) -> Morphology:
    morph = Morphology()
    morph.parameters.area = 1
    morph.parameters.closing_radius = 0
    for key, value in symmetry_kwargs.items():
        setattr(morph.parameters, key, value)
    return morph


def test_symmetry_flags_left_at_defaults_leave_the_mask_untouched():
    mask = _make_notched_mask()
    result = _morphology().Apply(mask)
    np.testing.assert_array_equal(result, mask)


def test_lateral_and_carves_out_both_sides_of_an_asymmetric_notch():
    mask = _make_notched_mask()
    morph = _morphology(symmetrize_lateral=True, symmetry_combine="and")

    result = morph.Apply(mask)

    assert not result[25, 42], "the notch itself has no mirrored counterpart"
    assert not result[25, 57], "AND should also carve out the notch's mirror on the intact side"
    assert result[70, 50], "the rest of the body is untouched"


def test_lateral_or_fills_in_the_notch_from_its_mirror():
    mask = _make_notched_mask()
    morph = _morphology(symmetrize_lateral=True, symmetry_combine="or")

    result = morph.Apply(mask)

    assert result[25, 42], "OR should fill the notch back in from its intact mirror"
    assert result[25, 57], "the mirror side stays filled"


def test_longitudinal_reflects_about_the_bounding_box_center_not_the_centroid():
    # Two blocks far apart along y (a big one and a small one), both
    # centered on the same x - the mass centroid (y ~ 76.2) sits well away
    # from the bounding-box center (y = 99.5) because the big block pulls
    # it up. Reflecting about the centroid instead of the bbox center would
    # place the mirrored copy of the small block ~47px off from where it
    # actually belongs.
    mask = np.zeros((200, 60), dtype=bool)
    mask[0:40, 20:40] = True
    mask[180:200, 20:40] = True

    morph = _morphology(symmetrize_longitudinal=True, symmetry_combine="or")
    result = morph.Apply(mask)

    probe_row, probe_col = 35, 30
    assert mask[probe_row, probe_col]  # sanity: probe is inside the big block
    assert result[164, probe_col], "mirrored about the bbox center (y=99.5), the probe lands here"
    assert not result[117, probe_col], "mirrored about the centroid (y=76.2) would have landed here instead"


def test_empty_mask_with_symmetry_enabled_is_a_safe_noop():
    mask = np.zeros((140, 100), dtype=bool)
    morph = _morphology(symmetrize_lateral=True, symmetrize_longitudinal=True)

    result = morph.Apply(mask)

    assert not result.any()
