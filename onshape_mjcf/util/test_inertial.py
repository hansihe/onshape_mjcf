import numpy as np
from onshape_mjcf.util.inertial import InertialData, TransformedInertialData, combineInertials

def test_combineInertials_single_inertial():
    inertial = TransformedInertialData(
        centroid=np.array([1, 2, 3]),
        mass=5.0,
        inertia=np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
        transform=np.eye(4)
    )
    result = combineInertials([inertial])
    assert result.mass == inertial.mass
    assert np.array_equal(result.centroid, inertial.centroid)
    assert np.array_equal(result.inertia, inertial.inertia)

def test_combineInertials_translation():
    inertial = TransformedInertialData(
        centroid=np.array([1, 2, 3]),
        mass=5.0,
        inertia=np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
        transform=np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    result = combineInertials([inertial])
    assert result.mass == 5.0
    assert np.allclose(result.centroid, np.array([2, 2, 3]))
    assert np.allclose(result.inertia, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))

def test_combineInertials_multiple_inertials():
    inertial1 = TransformedInertialData(
        centroid=np.array([1, 0, 0]),
        mass=4.0,
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        transform=np.eye(4)
    )
    inertial2 = TransformedInertialData(
        centroid=np.array([-2, 0, 0]),
        mass=4.0,
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        transform=np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    result = combineInertials([inertial1, inertial2])
    assert result.mass == 8.0
    assert np.allclose(result.centroid, np.array([0, 0, 0]))
    assert np.allclose(result.inertia, np.array([[2, 0, 0], [0, 10, 0], [0, 0, 10]]))

def test_combineInertials_multiple_inertials():
    inertial1 = TransformedInertialData(
        centroid=np.array([1, 1, 1]),
        mass=4.0,
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        transform=np.eye(4)
    )
    inertial2 = TransformedInertialData(
        centroid=np.array([-2, -2, -2]),
        mass=4.0,
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        transform=np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    )
    result = combineInertials([inertial1, inertial2])
    assert result.mass == 8.0
    assert np.allclose(result.centroid, np.array([0, 0, 0]))
    assert np.allclose(result.inertia, np.array([[18, -8, -8], [-8, 18, -8], [-8, -8, 18]]))

def test_combineInertials_rotation_transform():
    inertial = TransformedInertialData(
        centroid=np.array([1, 2, 3]),
        mass=5.0,
        inertia=np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
        transform=np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    result = combineInertials([inertial])
    assert result.mass == 5.0
    assert np.allclose(result.centroid, np.array([-1, -2, 3]))
    assert np.allclose(result.inertia, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))