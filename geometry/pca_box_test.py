import numpy as np

from geometry.pca_box import PCABox


def test_pca_box_to_local_places_min_corner_at_origin():
    points = np.array([[50, 50], [150, 50], [150, 200], [50, 200]], dtype=np.float32)
    box = PCABox(points)

    local_points = box.ToLocal(points)

    assert np.isclose(local_points.min(axis=0), [0, 0]).all()
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
    assert np.isclose(sorted(local_points.max(axis=0)), sorted([width, height])).all()
