"""
Tests for samurai Python bindings - MR Adaptation

Tests the make_MRAdapt function and MRAdapt callable object,
along with update_ghost_mr for mesh adaptation.
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


class TestMRAdaptCreation:
    """Tests for MRAdapt object creation."""

    def test_create_mr_adapt_1d(self):
        """Test creating MRAdapt for 1D field."""
        # Create mesh and field
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 0.0)

        # Create adaptation object
        MRadaptation = sam.make_MRAdapt(field)
        assert MRadaptation is not None
        assert type(MRadaptation).__name__ == "MRAdapt"

    def test_create_mr_adapt_2d(self):
        """Test creating MRAdapt for 2D field."""
        # Create mesh and field
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 5

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config)
        field = sam.ScalarField2D("u", mesh, 0.0)

        # Create adaptation object
        MRadaptation = sam.make_MRAdapt(field)
        assert MRadaptation is not None
        assert type(MRadaptation).__name__ == "MRAdapt"

    def test_create_mr_adapt_3d(self):
        """Test creating MRAdapt for 3D field."""
        # Create mesh and field
        config = sam.MeshConfig3D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.MRMesh3D(box, config)
        field = sam.ScalarField3D("u", mesh, 0.0)

        # Create adaptation object
        MRadaptation = sam.make_MRAdapt(field)
        assert MRadaptation is not None
        assert type(MRadaptation).__name__ == "MRAdapt"


class TestMRAdaptCallable:
    """Tests for MRAdapt as a callable object."""

    def test_mr_adapt_call_with_config_1d(self):
        """Test calling MRAdapt with config (1D)."""
        # Setup
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 0.0)

        # Create adaptation config
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2
        mra_config.regularity = 1.0

        # Create and call adaptation
        MRadaptation = sam.make_MRAdapt(field)
        MRadaptation(mra_config)  # Should not raise

    def test_mr_adapt_call_with_config_2d(self):
        """Test calling MRAdapt with config (2D)."""
        # Setup
        config_mesh = sam.MeshConfig2D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config_mesh)
        field = sam.ScalarField2D("u", mesh, 0.0)

        # Create adaptation config
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2

        # Create and call adaptation
        MRadaptation = sam.make_MRAdapt(field)
        MRadaptation(mra_config)  # Should not raise

    def test_mr_adapt_reusability(self):
        """Test that MRAdapt can be called multiple times."""
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 0.0)

        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2

        MRadaptation = sam.make_MRAdapt(field)

        # Call multiple times
        MRadaptation(mra_config)
        MRadaptation(mra_config)
        MRadaptation(mra_config)  # Should not raise


class TestUpdateGhostMr:
    """Tests for update_ghost_mr function."""

    def test_update_ghost_mr_1d(self):
        """Test update_ghost_mr for 1D field."""
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 0.0)

        # Should not raise
        sam.update_ghost_mr(field)

    def test_update_ghost_mr_2d(self):
        """Test update_ghost_mr for 2D field."""
        config_mesh = sam.MeshConfig2D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config_mesh)
        field = sam.ScalarField2D("u", mesh, 0.0)

        # Should not raise
        sam.update_ghost_mr(field)

    def test_update_ghost_mr_3d(self):
        """Test update_ghost_mr for 3D field."""
        config_mesh = sam.MeshConfig3D()
        config_mesh.min_level = 2
        config_mesh.max_level = 4

        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.MRMesh3D(box, config_mesh)
        field = sam.ScalarField3D("u", mesh, 0.0)

        # Should not raise
        sam.update_ghost_mr(field)


class TestAdaptationPipeline:
    """Tests for the complete adaptation pipeline."""

    def test_full_pipeline_1d(self):
        """Test complete pipeline: adapt + update ghosts (1D)."""
        # Setup
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 0.0)

        # Apply boundary conditions
        sam.make_dirichlet_bc(field, 0.0)

        # Create adaptation objects
        MRadaptation = sam.make_MRAdapt(field)
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2

        # Full pipeline
        MRadaptation(mra_config)
        sam.update_ghost_mr(field)

        # Should complete without errors

    def test_full_pipeline_2d(self):
        """Test complete pipeline: adapt + update ghosts (2D)."""
        # Setup
        config_mesh = sam.MeshConfig2D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config_mesh)
        field = sam.ScalarField2D("u", mesh, 0.0)

        # Apply boundary conditions
        sam.make_dirichlet_bc(field, 0.0)

        # Create adaptation objects
        MRadaptation = sam.make_MRAdapt(field)
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2

        # Full pipeline
        MRadaptation(mra_config)
        sam.update_ghost_mr(field)

        # Should complete without errors

    def test_iterative_adaptation(self):
        """Test multiple adaptation iterations."""
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 0.0)

        sam.make_dirichlet_bc(field, 0.0)

        MRadaptation = sam.make_MRAdapt(field)
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2

        # Multiple iterations (simulating time loop)
        for i in range(3):
            MRadaptation(mra_config)
            sam.update_ghost_mr(field)


class TestMRAConfigIntegration:
    """Tests for MRAConfig integration with MRAdapt."""

    def test_config_with_different_epsilon(self):
        """Test adaptation with different epsilon values."""
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 0.0)

        MRadaptation = sam.make_MRAdapt(field)

        # Test different epsilon values
        for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
            mra_config = sam.MRAConfig()
            mra_config.epsilon = eps
            MRadaptation(mra_config)  # Should not raise

    def test_config_with_different_regularity(self):
        """Test adaptation with different regularity values."""
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 0.0)

        MRadaptation = sam.make_MRAdapt(field)

        # Test different regularity values
        for reg in [0.0, 1.0, 2.0, 3.0]:
            mra_config = sam.MRAConfig()
            mra_config.regularity = reg
            MRadaptation(mra_config)  # Should not raise

    def test_config_with_relative_detail(self):
        """Test adaptation with relative_detail flag."""
        config_mesh = sam.MeshConfig1D()
        config_mesh.min_level = 2
        config_mesh.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config_mesh)
        field = sam.ScalarField1D("u", mesh, 1.0)  # Non-zero values

        MRadaptation = sam.make_MRAdapt(field)

        # Test with relative_detail
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2
        mra_config.relative_detail = True
        MRadaptation(mra_config)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
