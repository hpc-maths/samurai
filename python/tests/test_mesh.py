"""
Tests for samurai Python bindings - MRMesh class

Tests the samurai::MRMesh class bindings for 1D, 2D, and 3D.
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


class TestMRMesh1D:
    """Tests for MRMesh1D class."""

    def test_creation(self):
        """Test creating MRMesh1D from Box and MeshConfig."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.min_level = 0
        config.max_level = 1  # Keep small for testing

        mesh = sam.MRMesh1D(box, config)
        assert mesh.dim == 1

    def test_nb_cells(self):
        """Test nb_cells property and method."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.min_level = 0
        config.max_level = 1

        mesh = sam.MRMesh1D(box, config)
        # Total cells
        total = mesh.nb_cells()
        assert total > 0
        # Cells at specific level
        cells_level_0 = mesh.nb_cells(0)
        assert cells_level_0 > 0

    def test_level_properties(self):
        """Test min_level and max_level properties."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.min_level = 0
        config.max_level = 1

        mesh = sam.MRMesh1D(box, config)
        assert mesh.min_level == 0
        assert mesh.max_level == 1

    def test_graduation_width(self):
        """Test graduation_width property."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.graduation_width = 2

        mesh = sam.MRMesh1D(box, config)
        assert mesh.graduation_width == 2

    def test_ghost_width(self):
        """Test ghost_width property."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.max_stencil_radius = 2

        mesh = sam.MRMesh1D(box, config)
        assert mesh.ghost_width >= 1

    def test_max_stencil_radius(self):
        """Test max_stencil_radius property."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.max_stencil_radius = 3

        mesh = sam.MRMesh1D(box, config)
        assert mesh.max_stencil_radius == 3

    def test_cell_length(self):
        """Test cell_length method."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.min_level = 0
        config.max_level = 2

        mesh = sam.MRMesh1D(box, config)
        # Level 0 cells are larger
        len_0 = mesh.cell_length(0)
        len_1 = mesh.cell_length(1)
        len_2 = mesh.cell_length(2)
        assert len_0 > len_1 > len_2
        assert abs(len_0 - 1.0) < 1e-10
        assert abs(len_1 - 0.5) < 1e-10

    def test_min_cell_length(self):
        """Test min_cell_length property."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.min_level = 0
        config.max_level = 2

        mesh = sam.MRMesh1D(box, config)
        min_len = mesh.min_cell_length
        assert min_len > 0
        assert min_len <= mesh.cell_length(0)

    def test_periodicity_default(self):
        """Test default periodicity (non-periodic)."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()

        mesh = sam.MRMesh1D(box, config)
        assert not mesh.is_periodic()
        assert not mesh.is_periodic(0)
        assert mesh.periodicity == [False]

    def test_periodic_configuration(self):
        """Test setting periodic configuration."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.set_periodic(True)

        mesh = sam.MRMesh1D(box, config)
        assert mesh.is_periodic()
        assert mesh.is_periodic(0)

    def test_string_representation(self):
        """Test __repr__ and __str__."""
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.min_level = 0
        config.max_level = 1

        mesh = sam.MRMesh1D(box, config)
        repr_str = repr(mesh)
        str_str = str(mesh)

        assert "MRMesh1D" in repr_str
        assert "min_level=" in repr_str
        assert "L0-1" in str_str
        assert "cells" in str_str


class TestMRMesh2D:
    """Tests for MRMesh2D class."""

    def test_creation(self):
        """Test creating MRMesh2D from Box and MeshConfig."""
        box = sam.Box2D([0., 0.], [1., 1.])
        config = sam.MeshConfig2D()
        config.min_level = 0
        config.max_level = 1  # Keep small for testing

        mesh = sam.MRMesh2D(box, config)
        assert mesh.dim == 2

    def test_nb_cells(self):
        """Test nb_cells property and method."""
        box = sam.Box2D([0., 0.], [1., 1.])
        config = sam.MeshConfig2D()
        config.min_level = 0
        config.max_level = 1

        mesh = sam.MRMesh2D(box, config)
        total = mesh.nb_cells()
        assert total > 0
        cells_level_0 = mesh.nb_cells(0)
        assert cells_level_0 > 0

    def test_cell_length(self):
        """Test cell_length method in 2D."""
        box = sam.Box2D([0., 0.], [1., 1.])
        config = sam.MeshConfig2D()
        config.min_level = 0
        config.max_level = 1

        mesh = sam.MRMesh2D(box, config)
        len_0 = mesh.cell_length(0)
        len_1 = mesh.cell_length(1)
        assert len_0 > len_1
        assert abs(len_0 - 1.0) < 1e-10

    def test_periodicity_per_direction(self):
        """Test periodicity in each direction."""
        box = sam.Box2D([0., 0.], [1., 1.])
        config = sam.MeshConfig2D()
        config.set_periodic_per_direction([True, False])

        mesh = sam.MRMesh2D(box, config)
        assert mesh.is_periodic(0)
        assert not mesh.is_periodic(1)
        assert mesh.periodicity == [True, False]


class TestMRMesh3D:
    """Tests for MRMesh3D class."""

    def test_creation(self):
        """Test creating MRMesh3D from Box and MeshConfig."""
        box = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        config = sam.MeshConfig3D()
        config.min_level = 0
        config.max_level = 1  # Keep small for testing

        mesh = sam.MRMesh3D(box, config)
        assert mesh.dim == 3

    def test_nb_cells(self):
        """Test nb_cells for 3D mesh."""
        box = sam.Box3D([0., 0., 0.], [1., 1., 1.])
        config = sam.MeshConfig3D()
        config.min_level = 0
        config.max_level = 0  # Single level only

        mesh = sam.MRMesh3D(box, config)
        total = mesh.nb_cells()
        assert total > 0


class TestMRMeshSubmodule:
    """Tests for mesh submodule."""

    def test_mesh_submodule_exists(self):
        """Test that mesh submodule exists."""
        assert hasattr(sam, 'mesh')

    def test_mesh_classes_in_mesh(self):
        """Test that MRMesh classes are in mesh submodule."""
        ms = sam.mesh
        assert hasattr(ms, 'MRMesh1D')
        assert hasattr(ms, 'MRMesh2D')
        assert hasattr(ms, 'MRMesh3D')

    def test_mesh_from_submodule(self):
        """Test creating mesh from submodule."""
        MRMesh1D = sam.mesh.MRMesh1D
        box = sam.Box1D([0.], [1.])
        config = sam.MeshConfig1D()
        config.min_level = 0
        config.max_level = 1

        mesh = MRMesh1D(box, config)
        assert mesh.dim == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
