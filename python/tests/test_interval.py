"""
Tests for samurai Python bindings - Interval class

Tests the samurai::Interval class bindings.
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


class TestIntervalCreation:
    """Tests for Interval construction."""

    def test_default_constructor(self):
        """Test creating Interval with default constructor."""
        i = sam.Interval()
        assert i.start == 0
        assert i.end == 0
        assert i.step == 1
        assert i.index == 0

    def test_main_constructor(self):
        """Test creating Interval with start and end."""
        i = sam.Interval(5, 15)
        assert i.start == 5
        assert i.end == 15
        assert i.step == 1
        assert i.index == 0

    def test_constructor_with_index(self):
        """Test creating Interval with start, end, and index."""
        i = sam.Interval(5, 15, 100)
        assert i.start == 5
        assert i.end == 15
        assert i.step == 1
        assert i.index == 100

    def test_factory_function(self):
        """Test make_interval factory function."""
        i = sam.make_interval(0, 10)
        assert isinstance(i, sam.Interval)
        assert i.start == 0
        assert i.end == 10

    def test_factory_with_index(self):
        """Test make_interval with index parameter."""
        i = sam.make_interval(0, 10, index=5)
        assert i.start == 0
        assert i.end == 10
        assert i.index == 5


class TestIntervalProperties:
    """Tests for Interval property access."""

    def test_start_property(self):
        """Test start property getter and setter."""
        i = sam.Interval(0, 10)
        assert i.start == 0

        i.start = 5
        assert i.start == 5
        assert i.end == 10  # Other properties unchanged

    def test_end_property(self):
        """Test end property getter and setter."""
        i = sam.Interval(0, 10)
        assert i.end == 10

        i.end = 20
        assert i.start == 0
        assert i.end == 20

    def test_step_property(self):
        """Test step property getter and setter."""
        i = sam.Interval(0, 10)
        assert i.step == 1

        i.step = 2
        assert i.step == 2

    def test_index_property(self):
        """Test index property getter and setter."""
        i = sam.Interval(0, 10)
        assert i.index == 0

        i.index = 50
        assert i.index == 50


class TestIntervalQueryMethods:
    """Tests for Interval query methods."""

    def test_size(self):
        """Test size method."""
        i = sam.Interval(0, 10)
        assert i.size() == 10

        i = sam.Interval(5, 15)
        assert i.size() == 10

        i = sam.Interval(-5, 5)
        assert i.size() == 10

    def test_len(self):
        """Test len() function on Interval."""
        i = sam.Interval(0, 10)
        assert len(i) == 10

    def test_contains_true(self):
        """Test contains method for value in interval."""
        i = sam.Interval(0, 10)

        # Within interval
        assert i.contains(5) == True
        assert i.contains(0) == True
        assert i.contains(9) == True

        # Using 'in' operator
        assert 5 in i
        assert 0 in i
        assert 9 in i

    def test_contains_false(self):
        """Test contains method for value not in interval."""
        i = sam.Interval(0, 10)

        # Outside interval
        assert i.contains(-1) == False
        assert i.contains(10) == False  # end is exclusive
        assert i.contains(100) == False

        # Using 'in' operator
        assert -1 not in i
        assert 10 not in i
        assert 100 not in i

    def test_is_valid_true(self):
        """Test is_valid for non-empty interval."""
        i = sam.Interval(0, 10)
        assert i.is_valid() == True

        i = sam.Interval(-5, 5)
        assert i.is_valid() == True

    def test_is_valid_false(self):
        """Test is_valid for empty interval."""
        i = sam.Interval(5, 5)  # Empty interval
        assert i.is_valid() == False

        i = sam.Interval(10, 5)  # Invalid interval
        assert i.is_valid() == False

    def test_is_empty_true(self):
        """Test is_empty for empty interval."""
        i = sam.Interval(5, 5)
        assert i.is_empty() == True

    def test_is_empty_false(self):
        """Test is_empty for non-empty interval."""
        i = sam.Interval(0, 10)
        assert i.is_empty() == False


class TestIntervalElementSelection:
    """Tests for even/odd element selection."""

    def test_even_elements(self):
        """Test even_elements method."""
        i = sam.Interval(0, 10)
        even = i.even_elements()

        assert even.start == 0
        assert even.end == 9  # Last even (8) + 1
        assert even.step == 2

    def test_odd_elements(self):
        """Test odd_elements method."""
        i = sam.Interval(0, 10)
        odd = i.odd_elements()

        assert odd.start == 1
        assert odd.end == 10
        assert odd.step == 2


class TestIntervalArithmetic:
    """Tests for Interval arithmetic operators."""

    def test_multiply(self):
        """Test interval scaling multiplication."""
        i = sam.Interval(0, 10)
        scaled = i * 2

        assert scaled.start == 0
        assert scaled.end == 20
        assert scaled.step == 2

    def test_multiply_in_place(self):
        """Test in-place multiplication."""
        i = sam.Interval(0, 10)
        i *= 2

        assert i.start == 0
        assert i.end == 20

    def test_divide(self):
        """Test interval scaling division."""
        i = sam.Interval(0, 10)
        scaled = i / 2

        assert scaled.start == 0
        assert scaled.end == 5
        assert scaled.step == 1

    def test_right_shift_coarsen(self):
        """Test right shift (coarsening) operator."""
        i = sam.Interval(0, 10)
        coarsened = i >> 1

        # Coarsening divides by 2
        assert coarsened.start == 0
        assert coarsened.end == 5
        assert coarsened.step == 1

    def test_left_shift_refine(self):
        """Test left shift (refining) operator."""
        i = sam.Interval(0, 5)
        refined = i << 1

        # Refining multiplies by 2
        assert refined.start == 0
        assert refined.end == 10
        assert refined.step == 1

    def test_add_shift_right(self):
        """Test addition (shift right)."""
        i = sam.Interval(0, 10)
        shifted = i + 3

        assert shifted.start == 3
        assert shifted.end == 13

    def test_subtract_shift_left(self):
        """Test subtraction (shift left)."""
        i = sam.Interval(5, 15)
        shifted = i - 3

        assert shifted.start == 2
        assert shifted.end == 12


class TestIntervalComparison:
    """Tests for Interval comparison operators."""

    def test_equality_true(self):
        """Test equality for identical intervals."""
        i1 = sam.Interval(0, 10)
        i2 = sam.Interval(0, 10)

        assert i1 == i2
        assert not (i1 != i2)

    def test_equality_false(self):
        """Test inequality for different intervals."""
        i1 = sam.Interval(0, 10)
        i2 = sam.Interval(0, 15)

        assert i1 != i2
        assert not (i1 == i2)

    def test_less_than(self):
        """Test less than comparison (by start only)."""
        i1 = sam.Interval(0, 10)
        i2 = sam.Interval(5, 15)

        assert i1 < i2
        assert not (i2 < i1)


class TestIntervalStringRepresentation:
    """Tests for Interval string representations."""

    def test_repr(self):
        """Test __repr__ method."""
        i = sam.Interval(0, 10, index=5)
        repr_str = repr(i)

        assert "Interval" in repr_str
        assert "start=0" in repr_str
        assert "end=10" in repr_str
        assert "index=5" in repr_str

    def test_str(self):
        """Test __str__ method."""
        i = sam.Interval(0, 10)
        str_str = str(i)

        # Format is "[start,end[@index:step"
        assert "[0,10" in str_str
        assert "@0:1" in str_str


class TestIntervalSubmodule:
    """Tests for interval submodule."""

    def test_interval_submodule_exists(self):
        """Test that interval submodule exists."""
        assert hasattr(sam, 'interval')

    def test_interval_in_submodule(self):
        """Test that Interval is accessible from interval submodule."""
        iv = sam.interval
        assert hasattr(iv, 'Interval')

    def test_create_from_submodule(self):
        """Test creating Interval from submodule."""
        Interval = sam.interval.Interval
        i = Interval(0, 10)
        assert i.start == 0
        assert i.end == 10


class TestIntervalEdgeCases:
    """Tests for Interval edge cases."""

    def test_negative_coordinates(self):
        """Test Interval with negative coordinates."""
        i = sam.Interval(-5, 5)
        assert i.start == -5
        assert i.end == 5
        assert i.size() == 10

    def test_large_values(self):
        """Test Interval with large values."""
        i = sam.Interval(1000, 2000)
        assert i.size() == 1000
        assert i.contains(1500) == True

    def test_single_element_interval(self):
        """Test Interval with single element."""
        i = sam.Interval(5, 6)
        assert i.size() == 1
        assert i.contains(5) == True
        assert 6 not in i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
