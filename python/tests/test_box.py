"""
Tests for samurai Python bindings - Box class

Tests the samurai::Box<value_t, dim> class bindings for 1D, 2D, and 3D.
"""

import sys
import os
import numpy as np
import pytest

# Add the build directory to Python path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
except ImportError:
    pytest.skip("samurai_python module not built", allow_module_level=True)


class TestBox1D:
    """Tests for Box1D class."""

    def test_creation_from_list(self):
        """Test creating Box1D from Python lists."""
        box = sam.Box1D([0.], [1.])
        assert box.dim == 1

    def test_creation_from_numpy(self):
        """Test creating Box1D from numpy arrays."""
        min_corner = np.array([0.5])
        max_corner = np.array([2.5])
        box = sam.Box1D(min_corner, max_corner)
        assert box.dim == 1

    def test_corner_access(self):
        """Test accessing min_corner and max_corner."""
        box = sam.Box1D([0.], [1.])
        assert np.allclose(box.min_corner, [0.])
        assert np.allclose(box.max_corner, [1.])

    def test_corner_mutation(self):
        """Test modifying min_corner and max_corner."""
        box = sam.Box1D([0.], [1.])
        box.min_corner = np.array([0.5])
        box.max_corner = np.array([1.5])
        assert np.allclose(box.min_corner, [0.5])
        assert np.allclose(box.max_corner, [1.5])

    def test_length(self):
        """Test computing box length."""
        box = sam.Box1D([0.], [1.])
        length = box.length()
        assert np.allclose(length, [1.])

    def test_min_length(self):
        """Test computing minimum length."""
        box = sam.Box1D([0.], [1.])
        assert abs(box.min_length() - 1.0) < 1e-10

    def test_is_valid(self):
        """Test box validity check."""
        valid_box = sam.Box1D([0.], [1.])
        assert valid_box.is_valid()

        invalid_box = sam.Box1D([1.], [0.])
        assert not invalid_box.is_valid()

    def test_intersects(self):
        """Test box intersection check."""
        box_a = sam.Box1D([0.], [1.])
        box_b = sam.Box1D([0.5], [1.5])
        box_c = sam.Box1D([2.], [3.])

        assert box_a.intersects(box_b)
        assert not box_a.intersects(box_c)

    def test_intersection(self):
        """Test computing box intersection."""
        box_a = sam.Box1D([0.], [1.])
        box_b = sam.Box1D([0.5], [1.5])
        result = box_a.intersection(box_b)

        assert np.allclose(result.min_corner, [0.5])
        assert np.allclose(result.max_corner, [1.])

    def test_equality(self):
        """Test box equality operators."""
        box_a = sam.Box1D([0.], [1.])
        box_b = sam.Box1D([0.], [1.])
        box_c = sam.Box1D([0.], [2.])

        assert box_a == box_b
        assert box_a != box_c

    def test_scaling(self):
        """Test box scaling."""
        box = sam.Box1D([0.], [1.])
        scaled = box * 2.0
        assert np.allclose(scaled.length(), [2.])

        box *= 3.0
        assert np.allclose(box.length(), [3.])


class TestBox2D:
    """Tests for Box2D class."""

    def test_creation_from_list(self):
        """Test creating Box2D from Python lists."""
        box = sam.Box2D([0., 0.], [1., 1.])
        assert box.dim == 2

    def test_creation_from_numpy(self):
        """Test creating Box2D from numpy arrays."""
        min_corner = np.array([0.5, 0.5])
        max_corner = np.array([2.5, 2.5])
        box = sam.Box2D(min_corner, max_corner)
        assert box.dim == 2

    def test_corner_access(self):
        """Test accessing min_corner and max_corner."""
        box = sam.Box2D([0., 0.], [1., 1.])
        assert np.allclose(box.min_corner, [0., 0.])
        assert np.allclose(box.max_corner, [1., 1.])

    def test_corner_mutation(self):
        """Test modifying min_corner and max_corner."""
        box = sam.Box2D([0., 0.], [1., 1.])
        box.min_corner = np.array([0.5, 0.5])
        box.max_corner = np.array([1.5, 1.5])
        assert np.allclose(box.min_corner, [0.5, 0.5])
        assert np.allclose(box.max_corner, [1.5, 1.5])

    def test_length(self):
        """Test computing box length."""
        box = sam.Box2D([0., 0.], [1., 1.])
        length = box.length()
        assert np.allclose(length, [1., 1.])

    def test_asymmetric_box(self):
        """Test box with different dimensions."""
        box = sam.Box2D([0., 0.], [2., 1.])
        length = box.length()
        assert np.allclose(length, [2., 1.])
        assert abs(box.min_length() - 1.0) < 1e-10

    def test_is_valid(self):
        """Test box validity check."""
        valid_box = sam.Box2D([0., 0.], [1., 1.])
        assert valid_box.is_valid()

        invalid_box = sam.Box2D([1., 1.], [0., 0.])
        assert not invalid_box.is_valid()

    def test_intersects(self):
        """Test box intersection check."""
        box_a = sam.Box2D([0., 0.], [1., 1.])
        box_b = sam.Box2D([0.5, 0.5], [1.5, 1.5])
        box_c = sam.Box2D([2., 2.], [3., 3.])

        assert box_a.intersects(box_b)
        assert not box_a.intersects(box_c)

    def test_intersection(self):
        """Test computing box intersection."""
        box_a = sam.Box2D([0., 0.], [1., 1.])
        box_b = sam.Box2D([0.5, 0.5], [1.5, 1.5])
        result = box_a.intersection(box_b)

        assert np.allclose(result.min_corner, [0.5, 0.5])
        assert np.allclose(result.max_corner, [1., 1.])

    def test_equality(self):
        """Test box equality operators."""
        box_a = sam.Box2D([0., 0.], [1., 1.])
        box_b = sam.Box2D([0., 0.], [1., 1.])
        box_c = sam.Box2D([0., 0.], [1., 2.])

        assert box_a == box_b
        assert box_a != box_c

    def test_scaling(self):
        """Test box scaling."""
        box = sam.Box2D([0., 0.], [1., 1.])
        scaled = box * 2.0
        assert np.allclose(scaled.length(), [2., 2.])

        box *= 3.0
        assert np.allclose(box.length(), [3., 3.])


class TestBox3D:
    """Tests for Box3D class."""

    def test_creation_from_list(self):
        """Test creating Box3D from Python lists."""
        box = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        assert box.dim == 3

    def test_creation_from_numpy(self):
        """Test creating Box3D from numpy arrays."""
        min_corner = np.array([0.5, 0.5, 0.5])
        max_corner = np.array([2.5, 2.5, 2.5])
        box = sam.Box3D(min_corner, max_corner)
        assert box.dim == 3

    def test_corner_access(self):
        """Test accessing min_corner and max_corner."""
        box = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        assert np.allclose(box.min_corner, [0., 0., 0.])
        assert np.allclose(box.max_corner, [1., 1., 1.])

    def test_length(self):
        """Test computing box length."""
        box = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        length = box.length()
        assert np.allclose(length, [1., 1., 1.])

    def test_asymmetric_box(self):
        """Test box with different dimensions."""
        box = sam.Box3D([0., 0., 0.], [2., 1., 0.5])
        length = box.length()
        assert np.allclose(length, [2., 1., 0.5])
        assert abs(box.min_length() - 0.5) < 1e-10

    def test_is_valid(self):
        """Test box validity check."""
        valid_box = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        assert valid_box.is_valid()

        invalid_box = sam.Box3D([1., 1., 1.], [0., 0., 0.])
        assert not invalid_box.is_valid()

    def test_intersects(self):
        """Test box intersection check."""
        box_a = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        box_b = sam.Box3D([0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
        box_c = sam.Box3D([2., 2., 2.], [3., 3., 3.])

        assert box_a.intersects(box_b)
        assert not box_a.intersects(box_c)

    def test_scaling(self):
        """Test box scaling."""
        box = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        scaled = box * 2.0
        assert np.allclose(scaled.length(), [2., 2., 2.])

        box *= 3.0
        assert np.allclose(box.length(), [3., 3., 3.])


class TestBoxDifference:
    """Tests for Box::difference method."""

    def test_difference_non_intersecting(self):
        """Test difference when boxes don't intersect."""
        box_a = sam.Box2D([0., 0.], [1., 1.])
        box_b = sam.Box2D([2., 2.], [3., 3.])
        result = box_a.difference(box_b)
        # Should return original box since they don't intersect
        assert len(result) == 1
        assert result[0] == box_a

    def test_difference_intersecting(self):
        """Test difference when boxes intersect."""
        box_a = sam.Box2D([0., 0.], [2., 2.])
        box_b = sam.Box2D([1., 1.], [3., 3.])
        result = box_a.difference(box_b)
        # Should return multiple boxes
        assert len(result) > 0
        # All result boxes should be valid
        for box in result:
            assert box.is_valid()


class TestBoxGeometrySubmodule:
    """Tests for geometry submodule."""

    def test_geometry_submodule_exists(self):
        """Test that geometry submodule exists."""
        assert hasattr(sam, 'geometry')

    def test_box_classes_in_geometry(self):
        """Test that Box classes are accessible from geometry submodule."""
        geo = sam.geometry
        assert hasattr(geo, 'Box1D')
        assert hasattr(geo, 'Box2D')
        assert hasattr(geo, 'Box3D')

    def test_box_from_geometry(self):
        """Test creating Box from geometry submodule."""
        Box2D = sam.geometry.Box2D
        box = Box2D([0., 0.], [1., 1.])
        assert box.dim == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
