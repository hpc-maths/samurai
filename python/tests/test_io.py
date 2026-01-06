"""
Tests for samurai Python bindings - HDF5 I/O

Tests the save(), dump(), and load() functions for fields and meshes.
"""

import sys
import os
import pytest
import tempfile
import shutil

# Add the build directory to Python path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
except ImportError:
    pytest.skip("samurai_python module not built", allow_module_level=True)


class TestSaveFunction:
    """Tests for save() function (HDF5 + XDMF for Paraview)."""

    def test_save_1d_single_field(self):
        """Test saving 1D field with current directory."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 1.0)

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with path
            sam.save(tmpdir, "test_1d_save", field)

            # Check that files were created
            h5_file = os.path.join(tmpdir, "test_1d_save.h5")
            xdmf_file = os.path.join(tmpdir, "test_1d_save.xdmf")
            assert os.path.exists(h5_file), f"HDF5 file not created: {h5_file}"
            assert os.path.exists(xdmf_file), f"XDMF file not created: {xdmf_file}"

    def test_save_1d_none_path(self):
        """Test saving 1D field with None path (current directory)."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 2.0)

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Save with None path
                sam.save(None, "test_1d_none", field)

                # Check that files were created in current directory
                h5_file = "test_1d_none.h5"
                xdmf_file = "test_1d_none.xdmf"
                assert os.path.exists(h5_file), f"HDF5 file not created: {h5_file}"
                assert os.path.exists(xdmf_file), f"XDMF file not created: {xdmf_file}"
            finally:
                os.chdir(original_cwd)

    def test_save_2d_single_field(self):
        """Test saving 2D field."""
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config)
        field = sam.ScalarField2D("u", mesh, 3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.save(tmpdir, "test_2d_save", field)

            h5_file = os.path.join(tmpdir, "test_2d_save.h5")
            xdmf_file = os.path.join(tmpdir, "test_2d_save.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_save_3d_single_field(self):
        """Test saving 3D field."""
        config = sam.MeshConfig3D()
        config.min_level = 2
        config.max_level = 3

        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.MRMesh3D(box, config)
        field = sam.ScalarField3D("u", mesh, 4.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.save(tmpdir, "test_3d_save", field)

            h5_file = os.path.join(tmpdir, "test_3d_save.h5")
            xdmf_file = os.path.join(tmpdir, "test_3d_save.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_save_1d_two_fields(self):
        """Test saving 1D mesh with two fields."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field1 = sam.ScalarField1D("u", mesh, 1.0)
        field2 = sam.ScalarField1D("v", mesh, 2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.save(tmpdir, "test_1d_two", field1, field2)

            h5_file = os.path.join(tmpdir, "test_1d_two.h5")
            xdmf_file = os.path.join(tmpdir, "test_1d_two.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_save_1d_three_fields(self):
        """Test saving 1D mesh with three fields."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field1 = sam.ScalarField1D("u", mesh, 1.0)
        field2 = sam.ScalarField1D("v", mesh, 2.0)
        field3 = sam.ScalarField1D("w", mesh, 3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.save(tmpdir, "test_1d_three", field1, field2, field3)

            h5_file = os.path.join(tmpdir, "test_1d_three.h5")
            xdmf_file = os.path.join(tmpdir, "test_1d_three.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_save_filename_only_1d(self):
        """Test saving 1D field with filename only (current directory)."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Save with filename only
                sam.save("test_1d_file_only", field)

                h5_file = "test_1d_file_only.h5"
                xdmf_file = "test_1d_file_only.xdmf"
                assert os.path.exists(h5_file)
                assert os.path.exists(xdmf_file)
            finally:
                os.chdir(original_cwd)


class TestDumpFunction:
    """Tests for dump() function (HDF5-only for checkpoint/restart)."""

    def test_dump_1d_field(self):
        """Test dumping 1D field."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.dump(tmpdir, "test_1d_dump", field)

            # Check that only HDF5 file was created (no XDMF)
            h5_file = os.path.join(tmpdir, "test_1d_dump.h5")
            xdmf_file = os.path.join(tmpdir, "test_1d_dump.xdmf")
            assert os.path.exists(h5_file), f"HDF5 file not created: {h5_file}"
            assert not os.path.exists(xdmf_file), "XDMF file should not be created for dump"

    def test_dump_2d_field(self):
        """Test dumping 2D field."""
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config)
        field = sam.ScalarField2D("u", mesh, 2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.dump(tmpdir, "test_2d_dump", field)

            h5_file = os.path.join(tmpdir, "test_2d_dump.h5")
            assert os.path.exists(h5_file)

    def test_dump_3d_field(self):
        """Test dumping 3D field."""
        config = sam.MeshConfig3D()
        config.min_level = 2
        config.max_level = 3

        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.MRMesh3D(box, config)
        field = sam.ScalarField3D("u", mesh, 3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.dump(tmpdir, "test_3d_dump", field)

            h5_file = os.path.join(tmpdir, "test_3d_dump.h5")
            assert os.path.exists(h5_file)

    def test_dump_filename_only_1d(self):
        """Test dumping 1D field with filename only (current directory)."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 4.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                sam.dump("test_1d_dump_only", field)

                h5_file = "test_1d_dump_only.h5"
                assert os.path.exists(h5_file)
            finally:
                os.chdir(original_cwd)


class TestLoadFunction:
    """Tests for load() function (checkpoint/restart)."""

    def test_dump_load_1d_field(self):
        """Test dumping and loading 1D field."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 7.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Dump
            sam.dump(tmpdir, "test_1d_restart", field)

            # Create new mesh and field for loading
            mesh2 = sam.MRMesh1D(box, config)
            field2 = sam.ScalarField1D("u", mesh2, 0.0)

            # Load
            sam.load(tmpdir, "test_1d_restart", field2)

            # Check that the file still exists
            h5_file = os.path.join(tmpdir, "test_1d_restart.h5")
            assert os.path.exists(h5_file)

    def test_dump_load_2d_field(self):
        """Test dumping and loading 2D field."""
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config)
        field = sam.ScalarField2D("u", mesh, 8.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Dump
            sam.dump(tmpdir, "test_2d_restart", field)

            # Create new mesh and field for loading
            mesh2 = sam.MRMesh2D(box, config)
            field2 = sam.ScalarField2D("u", mesh2, 0.0)

            # Load
            sam.load(tmpdir, "test_2d_restart", field2)

            h5_file = os.path.join(tmpdir, "test_2d_restart.h5")
            assert os.path.exists(h5_file)

    def test_dump_load_3d_field(self):
        """Test dumping and loading 3D field."""
        config = sam.MeshConfig3D()
        config.min_level = 2
        config.max_level = 3

        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.MRMesh3D(box, config)
        field = sam.ScalarField3D("u", mesh, 9.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Dump
            sam.dump(tmpdir, "test_3d_restart", field)

            # Create new mesh and field for loading
            mesh2 = sam.MRMesh3D(box, config)
            field2 = sam.ScalarField3D("u", mesh2, 0.0)

            # Load
            sam.load(tmpdir, "test_3d_restart", field2)

            h5_file = os.path.join(tmpdir, "test_3d_restart.h5")
            assert os.path.exists(h5_file)

    def test_load_filename_only_1d(self):
        """Test loading 1D field with filename only (current directory)."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Dump with filename only
                sam.dump("test_1d_restart_only", field)

                # Create new mesh and field for loading
                mesh2 = sam.MRMesh1D(box, config)
                field2 = sam.ScalarField1D("u", mesh2, 0.0)

                # Load with filename only
                sam.load("test_1d_restart_only", field2)

                h5_file = "test_1d_restart_only.h5"
                assert os.path.exists(h5_file)
            finally:
                os.chdir(original_cwd)


class TestIoIntegration:
    """Integration tests for I/O functions with adaptation."""

    def test_adapt_save_pipeline_1d(self):
        """Test full pipeline: adapt + save (1D)."""
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 5

        box = sam.Box1D([0.0], [1.0])
        mesh = sam.MRMesh1D(box, config)
        field = sam.ScalarField1D("u", mesh, 1.0)

        # Apply boundary conditions
        sam.make_dirichlet_bc(field, 0.0)

        # Adaptation
        MRadaptation = sam.make_MRAdapt(field)
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2

        MRadaptation(mra_config)
        sam.update_ghost_mr(field)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            sam.save(tmpdir, "test_1d_adapt", field)

            h5_file = os.path.join(tmpdir, "test_1d_adapt.h5")
            xdmf_file = os.path.join(tmpdir, "test_1d_adapt.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_adapt_save_pipeline_2d(self):
        """Test full pipeline: adapt + save (2D)."""
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 5

        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        mesh = sam.MRMesh2D(box, config)
        field = sam.ScalarField2D("u", mesh, 1.0)

        sam.make_dirichlet_bc(field, 0.0)

        MRadaptation = sam.make_MRAdapt(field)
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-2

        MRadaptation(mra_config)
        sam.update_ghost_mr(field)

        with tempfile.TemporaryDirectory() as tmpdir:
            sam.save(tmpdir, "test_2d_adapt", field)

            h5_file = os.path.join(tmpdir, "test_2d_adapt.h5")
            xdmf_file = os.path.join(tmpdir, "test_2d_adapt.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
