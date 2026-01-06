"""
Tests for samurai Python bindings - for_each_cell function

Tests the for_each_cell algorithm that iterates over individual mesh cells.
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


class TestForEachCell1D:
    """Tests for for_each_cell with 1D mesh."""

    def test_1d_basic_iteration(self):
        """Test basic cell iteration in 1D."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4
        mesh = sam.MRMesh1D(box, config)

        # Collect cells
        cells = []

        def callback(cell):
            cells.append({
                'level': cell.level,
                'index': cell.index,
                'center': cell.center(),
                'length': cell.length
            })

        sam.for_each_cell(mesh, callback)

        # Verify we got some cells
        assert len(cells) > 0, "Should have at least one cell"

        # All centers should be 1-element tuples
        for cell_data in cells:
            center = cell_data['center']
            assert isinstance(center, tuple), f"Center should be tuple, got {type(center)}"
            assert len(center) == 1, f"Center should have 1 element for 1D, got {len(center)}"

    def test_1d_cell_properties(self):
        """Test that cells have correct properties."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        cell_data_list = []

        def callback(cell):
            center = cell.center()
            corner = cell.corner()
            cell_data_list.append({
                'level': cell.level,
                'index': cell.index,
                'length': cell.length,
                'center': center[0],
                'corner': corner[0],
            })

        sam.for_each_cell(mesh, callback)

        # All cells should be at level 3
        for data in cell_data_list:
            assert data['level'] == 3, f"Cell should be at level 3, got {data['level']}"
            assert data['length'] > 0, f"Cell length should be positive"
            assert data['center'] > data['corner'], "Center should be > corner"

    def test_1d_field_indexing(self):
        """Test that cell.index works for field indexing."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        # Create a field
        u = sam.ScalarField1D("u", mesh, 0.0)

        # Set values using cell.index
        def callback(cell):
            u[cell.index] = cell.level * 10.0

        sam.for_each_cell(mesh, callback)

        # Verify values were set correctly
        def verify_callback(cell):
            assert u[cell.index] == cell.level * 10.0, \
                f"Field value at index {cell.index} should be {cell.level * 10.0}, got {u[cell.index]}"

        sam.for_each_cell(mesh, verify_callback)

    def test_1d_center_values(self):
        """Test that center values are within domain bounds."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        centers = []

        def callback(cell):
            center = cell.center()
            centers.append(center[0])

        sam.for_each_cell(mesh, callback)

        # All centers should be within [0, 1]
        for c in centers:
            assert 0.0 <= c <= 1.0, f"Center {c} should be within [0, 1]"


class TestForEachCell2D:
    """Tests for for_each_cell with 2D mesh."""

    def test_2d_center_structure(self):
        """Test that 2D cell centers are 2-element tuples."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh2D(box, config)

        centers = []

        def callback(cell):
            centers.append(cell.center())

        sam.for_each_cell(mesh, callback)

        # All centers should be 2-element tuples
        for center in centers:
            assert isinstance(center, tuple), f"Center should be tuple, got {type(center)}"
            assert len(center) == 2, f"Center should have 2 elements for 2D, got {len(center)}"
            assert isinstance(center[0], (int, float)), "X coordinate should be numeric"
            assert isinstance(center[1], (int, float)), "Y coordinate should be numeric"

    def test_2d_cell_count(self):
        """Test that we get the expected number of cells."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh2D(box, config)

        count = [0]

        def callback(cell):
            count[0] += 1

        sam.for_each_cell(mesh, callback)

        # Should have cells
        assert count[0] > 0, "Should have cells in 2D mesh"
        # At level 2, should have 4x4 = 16 cells
        assert count[0] >= 16, f"Should have at least 16 cells at level 2, got {count[0]}"

    def test_2d_center_bounds(self):
        """Test that cell centers are within domain bounds."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh2D(box, config)

        centers = []

        def callback(cell):
            centers.append(cell.center())

        sam.for_each_cell(mesh, callback)

        # All centers should be within [0, 1] x [0, 1]
        for x, y in centers:
            assert 0.0 <= x <= 1.0, f"X coordinate {x} should be within [0, 1]"
            assert 0.0 <= y <= 1.0, f"Y coordinate {y} should be within [0, 1]"

    def test_2d_circular_initialization(self):
        """Test the circular initialization pattern from advection_2d."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 4
        config.max_level = 4
        mesh = sam.MRMesh2D(box, config)

        # Create field
        u = sam.ScalarField2D("u", mesh, 0.0)

        # Circular initial condition
        x_center, y_center = 0.3, 0.3
        radius = 0.2

        def callback(cell):
            x, y = cell.center()
            if ((x - x_center) * (x - x_center) + (y - y_center) * (y - y_center)) <= radius * radius:
                u[cell.index] = 1.0
            else:
                u[cell.index] = 0.0

        sam.for_each_cell(mesh, callback)

        # Verify some cells are set to 1 and some to 0
        values = []
        def collect_callback(cell):
            values.append(u[cell.index])

        sam.for_each_cell(mesh, collect_callback)

        assert 1.0 in values, "Should have some cells set to 1.0"
        assert 0.0 in values, "Should have some cells set to 0.0"


class TestForEachCell3D:
    """Tests for for_each_cell with 3D mesh."""

    def test_3d_center_structure(self):
        """Test that 3D cell centers are 3-element tuples."""
        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.MeshConfig3D()
        config.min_level = 1
        config.max_level = 1
        mesh = sam.MRMesh3D(box, config)

        centers = []

        def callback(cell):
            centers.append(cell.center())

        sam.for_each_cell(mesh, callback)

        # All centers should be 3-element tuples
        for center in centers:
            assert isinstance(center, tuple), f"Center should be tuple, got {type(center)}"
            assert len(center) == 3, f"Center should have 3 elements for 3D, got {len(center)}"

    def test_3d_corner_structure(self):
        """Test that corner() also returns 3-element tuples."""
        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.MeshConfig3D()
        config.min_level = 1
        config.max_level = 1
        mesh = sam.MRMesh3D(box, config)

        corners = []

        def callback(cell):
            corners.append(cell.corner())

        sam.for_each_cell(mesh, callback)

        # All corners should be 3-element tuples
        for corner in corners:
            assert isinstance(corner, tuple), f"Corner should be tuple, got {type(corner)}"
            assert len(corner) == 3, f"Corner should have 3 elements for 3D, got {len(corner)}"

    def test_3d_cell_count(self):
        """Test that we get cells in 3D."""
        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.MeshConfig3D()
        config.min_level = 1
        config.max_level = 1
        mesh = sam.MRMesh3D(box, config)

        count = [0]

        def callback(cell):
            count[0] += 1

        sam.for_each_cell(mesh, callback)

        # Should have cells
        assert count[0] > 0, "Should have cells in 3D mesh"


class TestForEachCellIntegration:
    """Integration tests for for_each_cell."""

    def test_cell_type_match(self):
        """Test that cells passed to callback are Cell instances."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        cell_types = set()

        def callback(cell):
            cell_types.add(type(cell).__name__)

        sam.for_each_cell(mesh, callback)

        # All cells should be Cell1D type
        assert 'Cell1D' in cell_types, f"Expected Cell1D type, got {cell_types}"

    def test_cell_repr(self):
        """Test that Cell has a string representation."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        reprs = []

        def callback(cell):
            reprs.append(repr(cell))

        sam.for_each_cell(mesh, callback)

        # All reprs should contain level and index info
        for r in reprs:
            assert 'Cell1D' in r, f"Repr should contain 'Cell1D', got {r}"
            assert 'level=' in r, f"Repr should contain 'level=', got {r}"

    def test_field_integration(self):
        """Test that for_each_cell works with field operations."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh2D(box, config)

        u = sam.ScalarField2D("u", mesh, 0.0)

        # Set field using cell centers
        def callback(cell):
            x, y = cell.center()
            u[cell.index] = x + y  # Simple function

        sam.for_each_cell(mesh, callback)

        # Verify some values
        count = [0]
        total = [0.0]

        def verify_callback(cell):
            x, y = cell.center()
            expected = x + y
            actual = u[cell.index]
            # Allow for small floating point errors
            assert abs(actual - expected) < 1e-10, \
                f"Field value mismatch: expected {expected}, got {actual}"
            count[0] += 1
            total[0] += actual

        sam.for_each_cell(mesh, verify_callback)

        assert count[0] > 0, "Should have verified some cells"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
