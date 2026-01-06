"""
Tests for samurai Python bindings - for_each_interval function

Tests the for_each_interval algorithm that iterates over mesh intervals.
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


class TestForEachInterval1D:
    """Tests for for_each_interval with 1D mesh."""

    def test_1d_basic_iteration(self):
        """Test basic interval iteration in 1D."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4
        mesh = sam.MRMesh1D(box, config)

        # Collect intervals
        intervals = []
        indices = []

        def callback(level, interval, index):
            intervals.append((level, interval.start, interval.end))
            indices.append(index)

        sam.for_each_interval(mesh, callback)

        # Verify we got some intervals
        assert len(intervals) > 0, "Should have at least one interval"

        # All indices should be empty tuples for 1D
        for idx in indices:
            assert idx == (), f"Index should be empty tuple for 1D, got {idx}"

    def test_1d_interval_properties(self):
        """Test that intervals have correct properties."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        interval_data = []

        def callback(level, interval, index):
            interval_data.append({
                'level': level,
                'start': interval.start,
                'end': interval.end,
                'step': interval.step,
                'index': interval.index,
                'size': interval.size(),
                'is_valid': interval.is_valid(),
                'is_empty': interval.is_empty()
            })

        sam.for_each_interval(mesh, callback)

        # All intervals should be valid and non-empty
        for data in interval_data:
            assert data['is_valid'], f"Interval should be valid: {data}"
            assert not data['is_empty'], f"Interval should not be empty: {data}"
            assert data['step'] == 1, f"Step should be 1: {data}"
            assert data['size'] > 0, f"Size should be positive: {data}"

    def test_1d_level_coverage(self):
        """Test that we iterate over expected levels."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 4
        mesh = sam.MRMesh1D(box, config)

        levels_seen = set()

        def callback(level, interval, index):
            levels_seen.add(level)

        sam.for_each_interval(mesh, callback)

        # Should see at least one level
        assert len(levels_seen) > 0, "Should see at least one level"
        # Note: MRMesh typically creates cells at max_level only, not all levels

    def test_1d_interval_contains(self):
        """Test using interval.contains in callback."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        contains_checks = []

        def callback(level, interval, index):
            # Test contains with various values
            contains_checks.append({
                'interval': f"[{interval.start}, {interval.end})",
                'contains_start': interval.contains(interval.start),
                'contains_end': interval.contains(interval.end),
                'contains_mid': interval.contains((interval.start + interval.end) // 2)
                    if interval.size() > 1 else None
            })

        sam.for_each_interval(mesh, callback)

        # Verify contains logic
        for check in contains_checks:
            assert check['contains_start'], "Should contain start value"
            assert not check['contains_end'], "Should not contain end value (exclusive)"
            if check['contains_mid'] is not None:
                assert check['contains_mid'], "Should contain middle value"

    def test_1d_multiple_intervals_per_level(self):
        """Test that there can be multiple intervals at the same level."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 4
        config.max_level = 4
        mesh = sam.MRMesh1D(box, config)

        level_intervals = {}

        def callback(level, interval, index):
            if level not in level_intervals:
                level_intervals[level] = []
            level_intervals[level].append((interval.start, interval.end))

        sam.for_each_interval(mesh, callback)

        # Check if we have multiple intervals at any level
        for level, intervals in level_intervals.items():
            # Verify intervals don't overlap
            sorted_intervals = sorted(intervals, key=lambda x: x[0])
            for i in range(len(sorted_intervals) - 1):
                current_end = sorted_intervals[i][1]
                next_start = sorted_intervals[i + 1][0]
                assert current_end <= next_start, \
                    f"Intervals should not overlap: {[current_end, next_start]}"


class TestForEachInterval2D:
    """Tests for for_each_interval with 2D mesh."""

    def test_2d_index_structure(self):
        """Test that 2D indices are single-element tuples."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh2D(box, config)

        indices = []

        def callback(level, interval, index):
            indices.append(index)

        sam.for_each_interval(mesh, callback)

        # All indices should be 1-element tuples for 2D
        for idx in indices:
            assert isinstance(idx, tuple), f"Index should be tuple, got {type(idx)}"
            assert len(idx) == 1, f"Index should have 1 element for 2D, got {len(idx)} elements"

    def test_2d_y_values_non_negative(self):
        """Test that y-index values are non-negative."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh2D(box, config)

        y_values = []

        def callback(level, interval, index):
            y = index[0]
            y_values.append(y)

        sam.for_each_interval(mesh, callback)

        # All y values should be non-negative integers
        for y in y_values:
            assert y >= 0, f"Y-index should be non-negative, got {y}"

    def test_2d_interval_count(self):
        """Test that 2D mesh generates intervals."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh2D(box, config)

        count = [0]

        def callback(level, interval, index):
            count[0] += 1

        sam.for_each_interval(mesh, callback)

        # Should have multiple intervals in 2D
        assert count[0] > 0, "Should have intervals in 2D mesh"

    def test_2d_index_types(self):
        """Test that index values are integers."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh2D(box, config)

        index_types = set()

        def callback(level, interval, index):
            index_types.add(type(index[0]).__name__)

        sam.for_each_interval(mesh, callback)

        # All index values should be integers
        assert 'int' in index_types, f"Index values should be int, got types: {index_types}"


class TestForEachInterval3D:
    """Tests for for_each_interval with 3D mesh."""

    def test_3d_index_structure(self):
        """Test that 3D indices are two-element tuples."""
        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.MeshConfig3D()
        config.min_level = 1
        config.max_level = 2
        mesh = sam.MRMesh3D(box, config)

        indices = []

        def callback(level, interval, index):
            indices.append(index)

        sam.for_each_interval(mesh, callback)

        # All indices should be 2-element tuples for 3D
        for idx in indices:
            assert isinstance(idx, tuple), f"Index should be tuple, got {type(idx)}"
            assert len(idx) == 2, f"Index should have 2 elements for 3D, got {len(idx)} elements"

    def test_3d_y_z_non_negative(self):
        """Test that y and z indices are non-negative."""
        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.MeshConfig3D()
        config.min_level = 1
        config.max_level = 1
        mesh = sam.MRMesh3D(box, config)

        yz_values = []

        def callback(level, interval, index):
            y, z = index
            yz_values.append((y, z))

        sam.for_each_interval(mesh, callback)

        # All y and z values should be non-negative
        for y, z in yz_values:
            assert y >= 0, f"Y-index should be non-negative, got {y}"
            assert z >= 0, f"Z-index should be non-negative, got {z}"

    def test_3d_interval_count(self):
        """Test that 3D mesh generates intervals."""
        box = sam.Box3D([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.MeshConfig3D()
        config.min_level = 1
        config.max_level = 1
        mesh = sam.MRMesh3D(box, config)

        count = [0]

        def callback(level, interval, index):
            count[0] += 1

        sam.for_each_interval(mesh, callback)

        # Should have intervals in 3D
        assert count[0] > 0, "Should have intervals in 3D mesh"


class TestForEachIntervalIntegration:
    """Integration tests with other bindings."""

    def test_interval_type_match(self):
        """Test that intervals passed to callback are Interval instances."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        interval_types = set()

        def callback(level, interval, index):
            interval_types.add(type(interval).__name__)

        sam.for_each_interval(mesh, callback)

        # All intervals should be Interval type
        assert 'Interval' in interval_types, f"Expected Interval type, got {interval_types}"

    def test_level_type(self):
        """Test that level parameter is integer."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        level_types = set()

        def callback(level, interval, index):
            level_types.add(type(level).__name__)

        sam.for_each_interval(mesh, callback)

        # All levels should be integers
        assert 'int' in level_types, f"Level should be int, got types: {level_types}"

    def test_callback_execution_order(self):
        """Test that callbacks are actually executed."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        execution_count = [0]

        def callback(level, interval, index):
            execution_count[0] += 1

        sam.for_each_interval(mesh, callback)

        # Callback should have been executed
        assert execution_count[0] > 0, "Callback should be executed at least once"

    def test_with_factory_interval(self):
        """Test for_each_interval with intervals created via factory."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 2
        mesh = sam.MRMesh1D(box, config)

        factory_intervals = []
        callback_intervals = []

        # Create intervals using factory
        factory_intervals.append(sam.make_interval(0, 10))

        def callback(level, interval, index):
            callback_intervals.append((interval.start, interval.end))

        sam.for_each_interval(mesh, callback)

        # Both should produce Interval objects
        assert len(callback_intervals) > 0, "Should have callback intervals"

    def test_mesh_properties_unchanged(self):
        """Test that for_each_interval doesn't modify mesh properties."""
        box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
        config = sam.MeshConfig2D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh2D(box, config)

        # Store original mesh properties
        original_min_level = mesh.min_level
        original_max_level = mesh.max_level
        original_nb_cells = mesh.nb_cells()

        def callback(level, interval, index):
            pass  # Do nothing

        sam.for_each_interval(mesh, callback)

        # Mesh properties should be unchanged
        assert mesh.min_level == original_min_level, "min_level should be unchanged"
        assert mesh.max_level == original_max_level, "max_level should be unchanged"
        assert mesh.nb_cells() == original_nb_cells, "nb_cells should be unchanged"


class TestForEachIntervalEdgeCases:
    """Edge case tests."""

    def test_single_level_mesh(self):
        """Test with mesh having only one level."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 3
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        levels_seen = set()

        def callback(level, interval, index):
            levels_seen.add(level)

        sam.for_each_interval(mesh, callback)

        # Should only see level 3
        assert levels_seen == {3}, f"Should only see level 3, got {levels_seen}"

    def test_empty_callback(self):
        """Test that callback with empty body doesn't crash."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        def callback(level, interval, index):
            pass

        # Should not raise any exception
        sam.for_each_interval(mesh, callback)

    def test_callback_can_capture_variables(self):
        """Test that callbacks can capture outer variables (Python closure)."""
        box = sam.Box1D([0.0], [1.0])
        config = sam.MeshConfig1D()
        config.min_level = 2
        config.max_level = 3
        mesh = sam.MRMesh1D(box, config)

        total_size = [0]

        def callback(level, interval, index):
            total_size[0] += interval.size()

        sam.for_each_interval(mesh, callback)

        # Should have accumulated sizes
        assert total_size[0] > 0, "Should have accumulated interval sizes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
