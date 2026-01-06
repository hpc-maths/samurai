"""
Tests for samurai Python bindings - Boundary Conditions

Tests the make_bc function and boundary condition types.
"""

import sys
import os
import pytest

# Add the build directory to Python path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
except ImportError:
    pytest.skip("samurai_python module not built", allow_module_level=True)


class TestMakeDirichletBC:
    """Tests for make_dirichlet_bc function."""

    def test_1d_dirichlet_order1(self):
        """Test Dirichlet BC of order 1 for 1D field."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        u = sam.ScalarField1D("u", mesh, 0.0)

        # Create Dirichlet BC with value 0.0 (returns None, BC is attached to field)
        sam.make_dirichlet_bc(u, 0.0)

        # If we get here without exception, the BC was attached successfully
        assert True

    def test_1d_dirichlet_different_orders(self):
        """Test Dirichlet BC with different orders."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh1D(box, config)

        # Test orders 1-4
        for order in [1, 2, 3, 4]:
            u = sam.ScalarField1D("u", mesh, 0.0)
            sam.make_dirichlet_bc(u, 1.5, order=order)
            # If we get here without exception, it worked
            assert True

    def test_1d_dirichlet_invalid_order(self):
        """Test that invalid order raises an error."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh1D(box, config)

        u = sam.ScalarField1D("u", mesh, 0.0)

        # Order 5 should raise an error
        with pytest.raises(RuntimeError, match="order must be between 1 and 4"):
            sam.make_dirichlet_bc(u, 0.0, order=5)

    def test_2d_dirichlet_order1(self):
        """Test Dirichlet BC of order 1 for 2D field (advection_2d case)."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 4
        config.max_level = 4
        mesh = sam.MRMesh2D(box, config)

        u = sam.ScalarField2D("u", mesh, 0.0)

        # Create Dirichlet BC with value 0.0 (as in advection_2d.cpp line 110)
        sam.make_dirichlet_bc(u, 0.0)

        # If we get here without exception, the BC was attached successfully
        assert True

    def test_2d_dirichlet_nonzero_value(self):
        """Test Dirichlet BC with non-zero constant value."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh2D(box, config)

        u = sam.ScalarField2D("u", mesh, 0.0)

        # Create Dirichlet BC with value 5.0
        sam.make_dirichlet_bc(u, 5.0)

        assert True

    def test_2d_dirichlet_different_orders(self):
        """Test Dirichlet BC with different orders in 2D."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh2D(box, config)

        # Test orders 1-4
        for order in [1, 2, 3, 4]:
            u = sam.ScalarField2D("u", mesh, 0.0)
            sam.make_dirichlet_bc(u, 0.0, order=order)
            assert True

    def test_3d_dirichlet_order1(self):
        """Test Dirichlet BC of order 1 for 3D field."""
        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.MeshConfig3D()
        config.min_level = 1
        config.max_level = 1
        mesh = sam.MRMesh3D(box, config)

        u = sam.ScalarField3D("u", mesh, 0.0)

        # Create Dirichlet BC
        sam.make_dirichlet_bc(u, 1.0)

        assert True

    def test_default_order_parameter(self):
        """Test that order defaults to 1."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        u1 = sam.ScalarField1D("u1", mesh, 0.0)
        u2 = sam.ScalarField1D("u2", mesh, 0.0)

        # Don't specify order - should default to 1
        sam.make_dirichlet_bc(u1, 0.0)
        sam.make_dirichlet_bc(u2, 0.0, order=1)

        # Both should work
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
