# Samurai Python Bindings: Strategy Analysis

## Executive Summary

After analyzing the Samurai C++ library through 8 independent strategy explorations, this document presents a comprehensive recommendation for creating Python bindings using pybind11.

**Key Finding**: The library's heavy template usage, expression template system, and multi-resolution mesh complexity require a **hybrid approach** combining code generation with manual wrappers.

---

## Strategy Comparison Matrix

| Strategy | Feasibility | Development Time | Maintenance | Performance | Pythonic Feel |
|----------|-------------|------------------|-------------|-------------|---------------|
| **Direct Minimal Wrappers** | Low (template issues) | 6-12 months | High | High | Low |
| **High-Level Pythonic Facade** | Medium | 4-6 months | Medium | Medium | **Very High** |
| **Field & Operations Wrapping** | Medium | 3-4 months | Medium | High | Medium |
| **Mesh & Adaptation API** | Medium | 2-3 months | Low | High | Medium |
| **Time Stepping & Solvers** | High | 2-3 months | Medium | High | Medium |
| **I/O and Checkpointing** | **High** | 1-2 months | Low | N/A | High |
| **Code Generation** | **Very High** | 2-3 months setup | Low | High | N/A |
| **Hybrid Layered** | **High** | 6-9 months | Low | **Very High** | High |

---

## Recommended Strategy: Hybrid Layered Architecture

Based on the analysis, I recommend a **3-layer hybrid architecture**:

### Layer 1: Generated Core Bindings (C++/pybind11)
**Purpose**: Handle template complexity automatically

**Components**:
- Auto-generated bindings for common template instantiations
- Mesh types: 1D, 2D, 3D with `double` value type
- Fields: `ScalarField` and `VectorField` (2-3 components)
- Core operators: diffusion, convection, identity
- Boundary conditions: Dirichlet, Neumann

**Implementation**: Use clang-based code generation to instantiate common combinations

### Layer 2: Manual Performance-Critical Bindings (C++/pybind11)
**Purpose**: Handle complex patterns that can't be generated

**Components**:
- Expression template evaluation
- PETSc solver integration
- AMR/MR adaptation workflow
- Cell iteration with Python callbacks
- Zero-copy NumPy integration

**Implementation**: Manual pybind11 wrappers with careful lifetime management

### Layer 3: Python Convenience Layer (Pure Python)
**Purpose**: Provide Pythonic high-level API

**Components**:
- Fluent mesh configuration API
- Time loop context managers
- Field initialization helpers
- I/O abstractions
- Visualization integration hooks

**Implementation**: Pure Python wrapping Layer 2

---

## Implementation Roadmap

### Phase 1: MVP Foundation (2 months)
**Goal**: Replicate basic heat equation demo

**Scope**:
1. Code generation setup for template instantiations
2. Bind `MRMesh<2>` and `ScalarField<MRMesh<2>, double>`
3. Bind diffusion operator
4. Bind Dirichlet/Neumann BCs (constant values)
5. HDF5 save/load
6. Python mesh configuration helper

**Milestones**:
- [ ] Generate field/mesh bindings for dim=1,2
- [ ] Create mesh configuration builder
- [ ] Bind diffusion operator
- [ ] Implement Python time loop
- [ ] Test: heat equation matches C++ output

### Phase 2: Core Features (2 months)
**Goal**: Replicate advection and convection demos

**New Features**:
1. Vector fields (2-3 components)
2. Convection operators (upwind, WENO5)
3. RK3 time stepping
4. Function-based boundary conditions
5. CFL-based adaptive time stepping

**Milestones**:
- [ ] Generate vector field bindings
- [ ] Bind convection schemes
- [ ] Implement RK3 integrator
- [ ] Python callable BC support
- [ ] Test: advection_2d matches C++ output

### Phase 3: Advanced Features (3 months)
**Goal**: Support AMR and implicit solvers

**New Features**:
1. AMR/MR adaptation interface
2. PETSc linear solver bindings
3. Implicit time stepping (backward Euler)
4. Multi-field adaptation
5. Checkpoint/restart capability

**Milestones**:
- [ ] Bind MRAdapt interface
- [ ] Bind PETSc KSP solvers
- [ ] Implement implicit stepper
- [ ] Checkpoint/save workflow
- [ ] Test: linear_convection_obstacle matches C++ output

### Phase 4: Polish & Optimization (2 months)
**Goal**: Production-ready API

**Tasks**:
1. Performance profiling and optimization
2. Comprehensive test suite
3. Documentation (tutorials, API reference)
4. Visualization integration
5. Packaging for pip installation

---

## Key Technical Decisions

### 1. Template Instantiation Strategy
**Decision**: Instantiate only common combinations

**Rationale**: Full combinatorial explosion = 144+ combinations. We only need:
- Dimensions: 1, 2, 3
- Value types: `double` (90% of cases)
- Components: 1, 2, 3
- Storage: xtensor (default)

**Result**: ~9 core instantiations, manageable manually or via generation

### 2. Expression Template Handling
**Decision**: Evaluate at C++/Python boundary, not lazy in Python

**Rationale**:
- Lazy evaluation in Python requires complex compute graph
- Eager evaluation matches common usage pattern
- C++ expression templates still work efficiently

**Implementation**:
```python
# Python: builds expression, evaluates immediately
unp1 = u - dt * diff(u)  # diff(u) is C++ call, subtraction in C++
```

### 3. Mesh Configuration API
**Decision**: Fluent builder pattern in Python

**Rationale**: Matches C++ API but more Pythonic than kwargs

**Example**:
```python
mesh = (samurai.MeshConfig(dim=2)
        .min_level(4)
        .max_level(8)
        .stencil_size(2)
        .build(box))
```

### 4. Field Data Access
**Decision**: Dual interface - cell-based and NumPy array

**Rationale**: Cell access for iteration, NumPy for vectorized operations

**Example**:
```python
# Cell-based (Python iteration)
for cell in mesh.cells():
    u[cell] = initial_condition(cell.center)

# NumPy-based (vectorized)
u_array = u.to_numpy()  # Zero-copy view
u_array[:] = np.exp(-x**2)
```

### 5. Time Stepping Architecture
**Decision**: Context managers for time loops

**Rationale**: Clean resource management, automatic checkpointing

**Example**:
```python
with samurai.TimeStepper(Tf=1.0, cfl=0.95, checkpoint_dir="results") as ts:
    for step in ts:
        mesh.adapt(u)
        u = u - ts.dt * convection(u)
        # Automatic saving at intervals
```

---

## File Structure

```
samurai/
├── CMakeLists.txt                 # Build configuration
├── include/
│   └── samurai_python/
│       ├── core.hpp                # Core C++ bindings
│       ├── fields.hpp              # Field wrappers
│       ├── operators.hpp           # Operator bindings
│       └── io.hpp                  # I/O bindings
├── python/
│   ├── samurai/
│   │   ├── __init__.py             # Public API
│   │   ├── mesh.py                 # Mesh configuration
│   │   ├── fields.py               # Field helpers
│   │   ├── schemes.py              # Scheme factories
│   │   ├── solvers.py              # Time integrators
│   │   ├── io.py                   # I/O interface
│   │   └── viz.py                  # Visualization hooks
│   └── generate/
│       ├── generate_bindings.py    # Code generator
│       └── templates/              # Binding templates
└── tests/
    ├── test_core.py                # C++ binding tests
    ├── test_api.py                 # Python API tests
    └── test_regression.py          # Compare vs C++
```

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Template instantiation issues | Medium | High | Use code generation, limit instantiations |
| Expression template binding | High | High | Force evaluation at boundary |
| PETSc solver complexity | High | High | Provide presets, expose advanced options |
| Memory leaks | Medium | High | Extensive testing, valgrind sanitizers |
| Performance overhead | Low | Medium | Profile, move critical paths to C++ |
| Build time increase | Medium | Low | Use precompiled headers, unity builds |

---

## Success Criteria

The binding strategy will be considered successful when:

1. **Functionality**: All FiniteVolume demos can be replicated in Python
2. **Performance**: Python version within 2x of C++ performance
3. **Usability**: Python API feels natural to NumPy/SciPy users
4. **Maintainability**: New C++ features require minimal binding work
5. **Testing**: Comprehensive test suite with >90% coverage
6. **Documentation**: Complete tutorial and API reference

---

## Next Steps

1. **Validate with stakeholders**: Review strategy with Samurai developers
2. **Prototype code generation**: Test clang-based binding generation
3. **Create MVP branch**: Start Phase 1 implementation
4. **Set up CI/CD**: Automated testing across platforms
5. **Begin documentation**: Tutorial parallel to implementation

---

## Appendix: Agent Findings Summary

### Agent 1: Direct Minimal Wrappers
- **Verdict**: Possible but with significant limitations
- **Key insight**: Expression templates cannot be directly exposed to Python
- **Estimated effort**: 3-6 months for basic functionality

### Agent 2: High-Level Pythonic Facade
- **Verdict**: Most user-friendly approach
- **Key insight**: Python-first design hides C++ complexity effectively
- **Estimated effort**: 4-6 months for full API

### Agent 3: Field & Operations Wrapping
- **Verdict**: Lazy evaluation preserves C++ efficiency
- **Key insight**: Hybrid eager/lazy evaluation recommended
- **Estimated effort**: 3-4 months for complete field API

### Agent 4: Mesh & Adaptation API
- **Verdict**: Lazy iteration is critical for performance
- **Key insight**: CellView pattern avoids materializing millions of cells
- **Estimated effort**: 2-3 months for mesh + adaptation

### Agent 5: Time Stepping & Solvers
- **Verdict**: Class-based for implicit, functional for explicit
- **Key insight**: Operator overloading mimics C++ syntax
- **Estimated effort**: 2-3 months for solvers

### Agent 6: I/O and Checkpointing
- **Verdict**: Straightforward HDF5 integration
- **Key insight**: SimulationIO context manager simplifies workflows
- **Estimated effort**: 1-2 months for I/O

### Agent 7: Code Generation Approach
- **Verdict**: Highly recommended and likely essential
- **Key insight**: 144+ template combinations require automation
- **Estimated effort**: 2-3 months setup, then automated

### Agent 8: Hybrid Layered Architecture
- **Verdict**: Best balance of performance and usability
- **Key insight**: 3-layer design enables incremental development
- **Estimated effort**: 6-9 months for complete system

---

**Document Version**: 1.0
**Date**: 2025-01-05
**Branch**: pybind11
**Status**: Awaiting Approval
