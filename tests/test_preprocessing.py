import numpy as np

from src.data.preprocessing import world_to_voxel, normalize_hu


def test_world_to_voxel():
    world = np.array([10.0, 20.0, 30.0])
    origin = np.array([0.0, 0.0, 0.0])
    spacing = np.array([1.0, 2.0, 5.0])

    voxel = world_to_voxel(world, origin, spacing)

    assert np.array_equal(voxel, np.array([10, 10, 6]))


def test_normalize_hu():
    volume = np.array([-1200, -1000, -300, 400, 1000])

    normalized = normalize_hu(volume)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized.dtype == np.float32

