"""
Test the geometric_smote module.
"""

import pytest
import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state

from ..geometric_smote import _make_geometric_sample, GeometricSMOTE

RANDOM_STATE = check_random_state(0)


@pytest.mark.parametrize('center,surface_point', [
    (RANDOM_STATE.random_sample((2, )), RANDOM_STATE.random_sample((2, ))),
    (2.6 * RANDOM_STATE.random_sample((4, )), 5.2 * RANDOM_STATE.random_sample((4, ))),
    (3.2 * RANDOM_STATE.random_sample((10, )), -3.5 * RANDOM_STATE.random_sample((10, ))),
    (-0.5 * RANDOM_STATE.random_sample((1, )), -10.9 * RANDOM_STATE.random_sample((1, )))
])
def test_make_geometric_sample_hypersphere(center, surface_point):
    """Test the generation of points inside a hypersphere."""
    point = _make_geometric_sample(center, surface_point, 0.0, 0.0, RANDOM_STATE)
    rel_point  = point - center
    rel_surface_point = surface_point - center
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))


@pytest.mark.parametrize('surface_point,deformation_factor', [
    (RANDOM_STATE.random_sample((2, )), 0.0),
    (2.6 * np.array([0.0, 1.0]), 0.25),
    (3.2 * np.array([0.0, 1.0, 0.0, 0.0]), 0.50),
    (0.5 * np.array([0.0, 0.0, 1.0]), 0.75),
    (6.7 * np.array([0.0, 0.0, 1.0, 0.0, 0.0]), 1.0),
])
def test_make_geometric_sample_half_hypersphere(surface_point, deformation_factor):
    """Test the generation of points inside a hypersphere."""
    center = np.zeros(surface_point.shape)
    point = _make_geometric_sample(center, surface_point, 1.0, deformation_factor, RANDOM_STATE)
    np.testing.assert_array_less(0.0, norm(surface_point) - norm(point))
    np.testing.assert_array_less(0.0, np.dot(point, surface_point))


@pytest.mark.parametrize('center,surface_point,truncation_factor', [
    (RANDOM_STATE.random_sample((2, )), RANDOM_STATE.random_sample((2, )), 0.0),
    (2.6 * RANDOM_STATE.random_sample((4, )), 5.2 * RANDOM_STATE.random_sample((4, )), 0.0),
    (3.2 * RANDOM_STATE.random_sample((10, )), -3.5 * RANDOM_STATE.random_sample((10, )), 0.0),
    (-0.5 * RANDOM_STATE.random_sample((1, )), -10.9 * RANDOM_STATE.random_sample((1, )), 0.0),
    (RANDOM_STATE.random_sample((2, )), RANDOM_STATE.random_sample((2, )), 1.0),
    (2.6 * RANDOM_STATE.random_sample((4, )), 5.2 * RANDOM_STATE.random_sample((4, )), 1.0),
    (3.2 * RANDOM_STATE.random_sample((10, )), -3.5 * RANDOM_STATE.random_sample((10, )), 1.0),
    (-0.5 * RANDOM_STATE.random_sample((1, )), -10.9 * RANDOM_STATE.random_sample((1, )), 1.0),
    (RANDOM_STATE.random_sample((2, )), RANDOM_STATE.random_sample((2, )), -1.0),
    (2.6 * RANDOM_STATE.random_sample((4, )), 5.2 * RANDOM_STATE.random_sample((4, )), -1.0),
    (3.2 * RANDOM_STATE.random_sample((10, )), -3.5 * RANDOM_STATE.random_sample((10, )), -1.0),
    (-0.5 * RANDOM_STATE.random_sample((1, )), -10.9 * RANDOM_STATE.random_sample((1, )), -1.0)
])
def test_make_geometric_sample_line_segment(center, surface_point, truncation_factor):
    """Test the generation of points on a line segment."""
    point = _make_geometric_sample(center, surface_point, truncation_factor, 1.0, RANDOM_STATE)
    rel_point  = point - center
    rel_surface_point = surface_point - center
    dot_product = np.dot(rel_point, rel_surface_point)
    norms_product = norm(rel_point) * norm(rel_surface_point)
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))
    dot_product = np.abs(dot_product) if truncation_factor == 0.0 else (-1) * dot_product
    np.testing.assert_allclose(np.abs(dot_product) / norms_product, 1.0)
