"""
Basic tests for Samurai Python bindings

These tests verify that the Python bindings can be imported and basic functionality works.
"""

import sys
import os

# Add the build directory to Python path for development
# In production, the module will be installed properly
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)


def test_module_import():
    """Test that the samurai_python module can be imported."""
    try:
        import samurai_python
        assert True, "Module imported successfully"
    except ImportError as e:
        # If module is not built yet, skip test
        import pytest
        pytest.skip(f"Module not built yet: {e}")


def test_version_attribute():
    """Test that the module has a __version__ attribute."""
    try:
        import samurai_python
        assert hasattr(samurai_python, "__version__")
        assert isinstance(samurai_python.__version__, str)
        assert len(samurai_python.__version__) > 0
    except ImportError:
        import pytest
        pytest.skip("Module not built yet")


def test_test_function():
    """Test the placeholder test_function."""
    try:
        import samurai_python
        result = samurai_python.test_function()
        assert result == "Samurai Python bindings are working!"
    except ImportError:
        import pytest
        pytest.skip("Module not built yet")
    except AttributeError:
        import pytest
        pytest.skip("test_function not yet implemented")


def test_module_docstring():
    """Test that the module has proper documentation."""
    try:
        import samurai_python
        assert samurai_python.__doc__ is not None
        assert len(samurai_python.__doc__) > 0
        assert "Samurai" in samurai_python.__doc__
    except ImportError:
        import pytest
        pytest.skip("Module not built yet")


if __name__ == "__main__":
    # Run tests manually for quick verification
    import pytest
    pytest.main([__file__, "-v"])
