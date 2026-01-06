#!/bin/bash
#
# Local test script for Python bindings CI/CD
# Mimics the GitHub Actions workflow locally
#
# Usage:
#   ./test_ci_local.sh              # Run all tests
#   ./test_ci_local.sh quick        # Run quick test only
#   ./test_ci_local.sh matrix       # Run test matrix
#   ./test_ci_local.sh demo         # Run demo validation
#   ./test_ci_local.sh check-nan    # Run CHECK_NAN test
#   ./test_ci_local.sh wheel        # Test wheel build
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR=${BUILD_DIR:-"build"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
BUILD_TYPE=${BUILD_TYPE:-"Release"}

echo -e "${GREEN}=== Samurai Python Bindings CI Local Test ===${NC}"
echo ""
echo "Configuration:"
echo "  BUILD_DIR: ${BUILD_DIR}"
echo "  PYTHON_VERSION: ${PYTHON_VERSION}"
echo "  BUILD_TYPE: ${BUILD_TYPE}"
echo ""

# Detect Python executable
PYTHON_EXE=$(which python3 || which python)
if [ -z "$PYTHON_EXE" ]; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: ${PYTHON_EXE}${NC}"
echo ""

#
# Function: print_section
#
print_section() {
    echo ""
    echo -e "${YELLOW}=== $1 ===${NC}"
    echo ""
}

#
# Function: check_dependencies
#
check_dependencies() {
    print_section "Checking Dependencies"

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}✗ CMake not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ CMake: $(cmake --version | head -n1)${NC}"

    # Check Ninja
    if command -v ninja &> /dev/null; then
        echo -e "${GREEN}✓ Ninja: $(ninja --version)${NC}"
        CMAKE_GENERATOR="-GNinja"
    else
        echo -e "${YELLOW}⚠ Ninja not found, using default generator${NC}"
        CMAKE_GENERATOR=""
    fi

    # Check Python packages
    ${PYTHON_EXE} -m pip show numpy &> /dev/null || {
        echo -e "${RED}✗ numpy not installed${NC}"
        echo "  Install with: pip install numpy"
        exit 1
    }
    echo -e "${GREEN}✓ numpy: $(${PYTHON_EXE} -c 'import numpy; print(numpy.__version__)')${NC}"

    ${PYTHON_EXE} -m pip show pytest &> /dev/null || {
        echo -e "${YELLOW}⚠ pytest not installed, installing...${NC}"
        ${PYTHON_EXE} -m pip install pytest
    }
    echo -e "${GREEN}✓ pytest: $(${PYTHON_EXE} -m pytest --version)${NC}"

    ${PYTHON_EXE} -m pip show h5py &> /dev/null || {
        echo -e "${YELLOW}⚠ h5py not installed, installing...${NC}"
        ${PYTHON_EXE} -m pip install h5py
    }
    echo -e "${GREEN}✓ h5py: $(${PYTHON_EXE} -c 'import h5py; print(h5py.__version__)')${NC}"
}

#
# Function: build_python_bindings
#
build_python_bindings() {
    print_section "Building Python Bindings"

    # Clean build directory if requested
    if [ "$CLEAN_BUILD" = "true" ]; then
        echo "Cleaning ${BUILD_DIR}..."
        rm -rf ${BUILD_DIR}
    fi

    # Configure
    echo "Configuring CMake..."
    cmake . \
        -B${BUILD_DIR} \
        ${CMAKE_GENERATOR} \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

    # Build
    echo "Building samurai_python..."
    cmake --build ${BUILD_DIR} --target samurai_python -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    echo -e "${GREEN}✓ Build complete${NC}"
}

#
# Function: test_import
#
test_import() {
    print_section "Testing Python Import"

    export PYTHONPATH="${PWD}/${BUILD_DIR}/python:${PYTHONPATH}"

    echo "Testing import..."
    ${PYTHON_EXE} -c "import samurai; print(f'✓ Samurai version: {samurai.__version__}')"

    echo "Testing available modules..."
    ${PYTHON_EXE} -c "
import samurai
print(f'✓ Available: {\", \".join([x for x in dir(samurai)[:10]])}...')
"
}

#
# Function: run_pytest
#
run_pytest() {
    print_section "Running Pytest Tests"

    export PYTHONPATH="${PWD}/${BUILD_DIR}/python:${PYTHONPATH}"

    cd python
    ${PYTHON_EXE} -m pytest tests/ -v --tb=short "$@"
    cd ..
}

#
# Function: run_demo
#
run_demo() {
    print_section "Running Demo Validation"

    export PYTHONPATH="${PWD}/${BUILD_DIR}/python:${PYTHONPATH}"

    # Install matplotlib if needed
    ${PYTHON_EXE} -m pip show matplotlib &> /dev/null || {
        echo "Installing matplotlib..."
        ${PYTHON_EXE} -m pip install matplotlib
    }

    # Run demo
    echo "Running advection_2d.py..."
    ${PYTHON_EXE} python/examples/advection_2d.py \
        --max_level 4 \
        --Tf 0.01 \
        --nfiles 3

    # Verify output
    echo ""
    echo "Verifying output files..."
    if ls FV_advection_2d_*.h5 1> /dev/null 2>&1; then
        echo -e "${GREEN}✓ HDF5 files created${NC}"
        ls -lh FV_advection_2d_*.h5
    else
        echo -e "${RED}✗ No HDF5 files created${NC}"
        exit 1
    fi
}

#
# Function: test_check_nan
#
test_check_nan() {
    print_section "Testing CHECK_NAN Mode"

    # Rebuild in Debug mode
    echo "Building with SAMURAI_CHECK_NAN=ON..."
    cmake . \
        -B${BUILD_DIR}-debug \
        ${CMAKE_GENERATOR} \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSAMURAI_CHECK_NAN=ON

    cmake --build ${BUILD_DIR}-debug --target samurai_python -j4

    # Run subset of tests
    export PYTHONPATH="${PWD}/${BUILD_DIR}-debug/python:${PYTHONPATH}"
    cd python
    ${PYTHON_EXE} -m pytest tests/ -v --tb=short -k "test_mesh or test_field or test_box"
    cd ..

    echo -e "${GREEN}✓ CHECK_NAN tests passed${NC}"
}

#
# Function: test_wheel_build
#
test_wheel_build() {
    print_section "Testing Wheel Build"

    # Check if build is installed
    ${PYTHON_EXE} -m pip show build &> /dev/null || {
        echo "Installing build tool..."
        ${PYTHON_EXE} -m pip install build
    }

    # Build wheel
    echo "Building wheel..."
    cd python
    ${PYTHON_EXE} -m build --wheel --outdir ../dist/
    cd ..

    # Display wheel
    echo ""
    echo "Built wheel:"
    ls -lh dist/*.whl || {
        echo -e "${RED}✗ Wheel build failed${NC}"
        exit 1
    }

    # Test wheel installation
    echo ""
    echo "Testing wheel installation..."
    ${PYTHON_EXE} -m pip uninstall samurai -y 2>/dev/null || true
    ${PYTHON_EXE} -m pip install dist/samurai*.whl

    # Verify
    ${PYTHON_EXE} -c "import samurai; print(f'✓ Installed samurai {samurai.__version__}')"

    # Run tests
    cd python
    ${PYTHON_EXE} -m pytest tests/ -v --tb=short
    cd ..

    echo -e "${GREEN}✓ Wheel test complete${NC}"
}

#
# Function: cleanup
#
cleanup() {
    print_section "Cleanup"

    echo "Removing build artifacts..."
    rm -rf ${BUILD_DIR}
    rm -rf ${BUILD_DIR}-debug
    rm -rf dist
    rm -f FV_advection_2d_*.h5

    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

#
# Main execution
#
main() {
    # Parse arguments
    TEST_TYPE=${1:-all}

    case $TEST_TYPE in
        quick)
            check_dependencies
            build_python_bindings
            test_import
            run_pytest -k "test_box or test_mesh_config"
            ;;
        matrix)
            check_dependencies
            build_python_bindings
            test_import
            run_pytest
            ;;
        demo)
            check_dependencies
            build_python_bindings
            run_demo
            ;;
        check-nan)
            check_dependencies
            test_check_nan
            ;;
        wheel)
            check_dependencies
            test_wheel_build
            ;;
        all)
            check_dependencies
            build_python_bindings
            test_import
            run_pytest
            run_demo
            ;;
        clean)
            cleanup
            ;;
        *)
            echo "Usage: $0 [quick|matrix|demo|check-nan|wheel|all|clean]"
            echo ""
            echo "Options:"
            echo "  quick      - Run quick test only"
            echo "  matrix     - Run full pytest suite"
            echo "  demo       - Run demo validation"
            echo "  check-nan  - Test CHECK_NAN mode"
            echo "  wheel      - Test wheel build"
            echo "  all        - Run all tests (default)"
            echo "  clean      - Clean build artifacts"
            exit 1
            ;;
    esac

    echo ""
    echo -e "${GREEN}=== All tests passed! ===${NC}"
}

# Run main
main "$@"
