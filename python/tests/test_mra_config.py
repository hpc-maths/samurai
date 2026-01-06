"""
Tests for samurai Python bindings - MRA Configuration

Tests the MRAConfig class and its properties for multiresolution adaptation.
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


class TestMRAConfigCreation:
    """Tests for MRAConfig creation and default values."""

    def test_default_creation(self):
        """Test creating MRAConfig with default values."""
        config = sam.MRAConfig()

        assert config is not None

    def test_default_epsilon(self):
        """Test that default epsilon is 1e-4."""
        config = sam.MRAConfig()
        assert config.epsilon == 1e-4

    def test_default_regularity(self):
        """Test that default regularity is 1.0."""
        config = sam.MRAConfig()
        assert config.regularity == 1.0

    def test_default_relative_detail(self):
        """Test that default relative_detail is False."""
        config = sam.MRAConfig()
        assert config.relative_detail is False


class TestMRAConfigProperties:
    """Tests for MRAConfig property setters."""

    def test_set_epsilon(self):
        """Test setting epsilon property."""
        config = sam.MRAConfig()
        config.epsilon = 2e-4
        assert config.epsilon == 2e-4

    def test_set_regularity(self):
        """Test setting regularity property."""
        config = sam.MRAConfig()
        config.regularity = 2.0
        assert config.regularity == 2.0

    def test_set_relative_detail_true(self):
        """Test setting relative_detail to True."""
        config = sam.MRAConfig()
        config.relative_detail = True
        assert config.relative_detail is True

    def test_set_relative_detail_false(self):
        """Test setting relative_detail to False."""
        config = sam.MRAConfig()
        config.relative_detail = True
        config.relative_detail = False
        assert config.relative_detail is False

    def test_multiple_properties(self):
        """Test setting multiple properties."""
        config = sam.MRAConfig()
        config.epsilon = 1e-3
        config.regularity = 3.0
        config.relative_detail = True

        assert config.epsilon == 1e-3
        assert config.regularity == 3.0
        assert config.relative_detail is True


class TestMRAConfigMethodChaining:
    """Tests for MRAConfig fluent interface using property setters."""

    def test_epsilon_property(self):
        """Test setting epsilon property."""
        config = sam.MRAConfig()
        config.epsilon = 2e-4
        assert config.epsilon == 2e-4

    def test_regularity_property(self):
        """Test setting regularity property."""
        config = sam.MRAConfig()
        config.regularity = 2.0
        assert config.regularity == 2.0

    def test_relative_detail_property(self):
        """Test setting relative_detail property."""
        config = sam.MRAConfig()
        config.relative_detail = True
        assert config.relative_detail is True

    def test_sequential_property_setting(self):
        """Test setting multiple properties sequentially."""
        config = sam.MRAConfig()
        config.epsilon = 1e-3
        config.regularity = 2.0
        config.relative_detail = True

        assert config.epsilon == 1e-3
        assert config.regularity == 2.0
        assert config.relative_detail is True

    def test_property_order_independence(self):
        """Test that property setting order doesn't matter."""
        config1 = sam.MRAConfig()
        config1.epsilon = 2e-4
        config1.regularity = 2.0

        config2 = sam.MRAConfig()
        config2.regularity = 2.0
        config2.epsilon = 2e-4

        assert config1.epsilon == config2.epsilon
        assert config1.regularity == config2.regularity


class TestMRAConfigStringRepresentation:
    """Tests for MRAConfig string representations."""

    def test_repr(self):
        """Test __repr__ string representation."""
        config = sam.MRAConfig()
        config.epsilon = 2e-4
        config.regularity = 2.0

        repr_str = repr(config)
        assert "MRAConfig" in repr_str
        assert "epsilon" in repr_str
        assert "regularity" in repr_str

    def test_str(self):
        """Test __str__ string representation."""
        config = sam.MRAConfig()
        str_str = str(config)
        assert "MRAConfig" in str_str


class TestMRAConfigEquality:
    """Tests for MRAConfig equality comparison."""

    def test_equal_default(self):
        """Test that two default configs are equal."""
        config1 = sam.MRAConfig()
        config2 = sam.MRAConfig()
        assert config1 == config2

    def test_equal_same_values(self):
        """Test equality with same custom values."""
        config1 = sam.MRAConfig()
        config1.epsilon = 2e-4
        config1.regularity = 2.0

        config2 = sam.MRAConfig()
        config2.epsilon = 2e-4
        config2.regularity = 2.0

        assert config1 == config2

    def test_not_equal_epsilon(self):
        """Test inequality with different epsilon."""
        config1 = sam.MRAConfig()
        config1.epsilon = 1e-4

        config2 = sam.MRAConfig()
        config2.epsilon = 2e-4

        assert config1 != config2

    def test_not_equal_regularity(self):
        """Test inequality with different regularity."""
        config1 = sam.MRAConfig()
        config1.regularity = 1.0

        config2 = sam.MRAConfig()
        config2.regularity = 2.0

        assert config1 != config2

    def test_not_equal_relative_detail(self):
        """Test inequality with different relative_detail."""
        config1 = sam.MRAConfig()
        config1.relative_detail = False

        config2 = sam.MRAConfig()
        config2.relative_detail = True

        assert config1 != config2


class TestMRAConfigTypicalValues:
    """Tests with typical values from real usage."""

    def test_advection_2d_values(self):
        """Test values from advection_2d.cpp demo."""
        config = sam.MRAConfig()
        config.epsilon = 2e-4
        assert config.epsilon == 2e-4
        assert config.regularity == 1.0  # default
        assert config.relative_detail is False  # default

    def test_fine_adaptation(self):
        """Test fine adaptation (low epsilon)."""
        config = sam.MRAConfig()
        config.epsilon = 1e-5
        assert config.epsilon == 1e-5

    def test_coarse_adaptation(self):
        """Test coarse adaptation (high epsilon)."""
        config = sam.MRAConfig()
        config.epsilon = 1e-1
        assert config.epsilon == 1e-1

    def test_minimal_gradation(self):
        """Test minimal gradation (zero regularity)."""
        config = sam.MRAConfig()
        config.regularity = 0.0
        assert config.regularity == 0.0

    def test_smooth_gradation(self):
        """Test smooth gradation (high regularity)."""
        config = sam.MRAConfig()
        config.regularity = 3.0
        assert config.regularity == 3.0


class TestMRAConfigReuse:
    """Tests for reusing configuration objects."""

    def test_config_reuse(self):
        """Test that config can be modified multiple times."""
        config = sam.MRAConfig()
        config.epsilon = 2e-4
        assert config.epsilon == 2e-4

        # Modify again
        config.epsilon = 1e-3
        assert config.epsilon == 1e-3

    def test_config_independence(self):
        """Test that independent configs don't affect each other."""
        config1 = sam.MRAConfig()
        config1.epsilon = 1e-4

        config2 = sam.MRAConfig()
        config2.epsilon = 2e-4

        config1.regularity = 2.0

        assert config1.regularity == 2.0
        assert config2.regularity == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
