# Samurai V2: Integrated Roadmap - Python Ecosystem & DSL Synergy

**Version:** 1.0
**Date:** 2025-01-05
**Status:** Strategic Integration Plan

---

## Executive Summary

Ce document présente la **vision intégrée** combinant les propositions du document 05 (Python Ecosystem) avec une couche DSL (Domain-Specific Language) en une stratégie cohérente pour l'évolution de Samurai V2. L'objectif est de créer un **continuum d'abstraction** permettant aux utilisateurs de progresser du niveau débutant au niveau expert tout en maintenant des performances optimales.

---

## 1. Vision Stratégique Unifiée

### 1.1 Philosophie: "Three-Layer Architecture"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SAMURAI V2 UNIFIED ECOSYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   LAYER 1: EQUATION LAYER (DSL)                      │  │
│  │                   Target: Mathematicians, Students                   │  │
│  │                   Entry Point: Mathematical Notation                 │  │
│  │                                                                        │  │
│  │   ∂u/∂t = D∇²u  ────────────▶  Samurai-DSL  ─────────▶  C++20 Code  │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                LAYER 2: INTERACTIVE LAYER (Python)                   │  │
│  │                Target: Researchers, Data Scientists                  │  │
│  │                Entry Point: Python/Jupyter                            │  │
│  │                                                                        │  │
│  │   import samurai as sam ─────────▶ Python API ──────▶  C++ Core       │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                  LAYER 3: PRODUCTION LAYER (C++20)                   │  │
│  │                  Target: HPC Experts, Production Engineers            │  │
│  │                  Entry Point: Direct C++ Development                  │  │
│  │                                                                        │  │
│  │   Explicit Control ───────────▶ Native C++20 ──────▶  Max Performance │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                              Shared C++20 Core                               │
│                         (AMR, Schemes, I/O, MPI, GPU)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 User Journey Matrix

| User Profile | Entry Point | Primary Tool | Exit Point | Typical Use Case |
|--------------|-------------|--------------|------------|------------------|
| **Undergraduate Student** | Layer 1 (DSL) | Jupyter Notebook | Layer 2 (Python) | Course projects, learning |
| **Graduate Researcher** | Layer 2 (Python) | Python API | Layer 3 (C++) | Prototyping, ML integration |
| **Data Scientist** | Layer 2 (Python) | JAX/PyTorch | Layer 2 only | Inverse problems, PINNs |
| **HPC Engineer** | Layer 3 (C++) | Native C++ | Layer 3 only | Production, supercomputing |
| **Computational Scientist** | Layer 1 (DSL) | Generated C++ | Layer 3 (C++) | Rapid iteration + optimization |

---

## 2. Integration Architecture

### 2.1 Shared Components

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SHARED INFRASTRUCTURE LAYER                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │ SymPy Parser   │  │ IR System      │  │ Code Templates │               │
│  │ (Common)       │  │ (Common)       │  │ (DSL-specific) │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│           │                   │                    │                        │
│           └───────────────────┴────────────────────┘                        │
│                              │                                               │
│                              ▼                                               │
│                    ┌──────────────────┐                                     │
│                    │  Type Database   │                                     │
│                    │  - Mesh types    │                                     │
│                    │  - Field types   │                                     │
│                    │  - BC types      │                                     │
│                    │  - Scheme types  │                                     │
│                    └──────────────────┘                                     │
│                              │                                               │
│                              ▼                                               │
│                    ┌──────────────────┐                                     │
│                    │ C++20 Reflection │                                     │
│                    │ & Metadata       │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Code Generation Flows

#### Flow 1: DSL → C++20 (Direct)

```
LaTeX Equation
       │
       ▼
SymPy Parser
       │
       ▼
PDESystem (IR)
       │
       ├─────────────────┐
       ▼                 ▼
Type Checker    Scheme Selector
       │                 │
       └─────────────────┘
              │
              ▼
Jinja2 Templates
              │
              ▼
C++20 Source File
              │
              ▼
Compiled Binary
```

#### Flow 2: DSL → Python API (Hybrid)

```
LaTeX Equation
       │
       ▼
SymPy Parser
       │
       ▼
PDESystem (IR)
       │
       ├─────────────────────┐
       ▼                     ▼
pybind11 Bindings   JAX Primitives
       │                     │
       └─────────────────────┘
                   │
                   ▼
        Python Module (.so/.pyd)
                   │
                   ▼
        Interactive Execution
```

#### Flow 3: Pure Python (Direct)

```
Python Code
       │
       ▼
Python API
       │
       ├─────────────────┐
       ▼                 ▼
pybind11 Layer   NumPy/JAX Bridge
       │                 │
       └─────────────────┘
                  │
                  ▼
C++20 Core (Runtime)
                  │
                  ▼
Results (NumPy arrays)
```

---

## 3. Component Mapping

### 3.1 DSL Components → C++ Implementation

| DSL Concept | C++20 Core Location | Implementation Status |
|-------------|---------------------|----------------------|
| `Box([0,0], [1,1])` | `samurai::Box<double, dim>` | ✅ Complete |
| `mesh_config(dim, min_level, max_level)` | `samurai::mesh_config<dim>` | ✅ Complete |
| `ScalarField("u", mesh)` | `samurai::make_scalar_field<double>` | ✅ Complete |
| `DirichletBC(u, 0)` | `samurai::make_bc<samurai::Dirichlet<1>>` | ✅ Complete |
| `for_each_cell(mesh, func)` | `samurai::for_each_cell` | ✅ Complete |
| `upwind(velocity, u)` | `samurai::upwind` | ✅ Complete |
| `adapt(mesh, criterion, epsilon)` | `samurai::make_MRAdapt` | ✅ Complete |

### 3.2 Python Bindings Priority Matrix

| Priority | Component | Complexity | User Value | Dependencies |
|----------|-----------|------------|------------|--------------|
| **P0** | Mesh types | Low | Critical | None |
| **P0** | Scalar/Vector fields | Low | Critical | Mesh |
| **P0** | for_each_cell | Low | Critical | Mesh, Field |
| **P1** | Boundary conditions | Medium | High | Field |
| **P1** | Operators (upwind, diffusion) | Medium | High | Field |
| **P2** | Adaptation | High | High | Field, MR |
| **P2** | I/O (HDF5) | Medium | Medium | Field |
| **P3** | PETSc solvers | High | Medium | Field, MPI |

---

## 4. Implementation Timeline (18 Months)

### Phase 1: Foundation (Months 1-6)

**Goal:** Create shared infrastructure and initial Python bindings

```
Month 1-2: Project Setup
├── samurai-dsl Python package structure
├── pybind11 integration in CMake
└── CI/CD pipeline for Python testing

Month 3-4: Core Type Bindings
├── Mesh bindings (Uniform, MR)
├── Field bindings (Scalar, Vector)
├── Basic algorithms (for_each_cell, for_each_interval)
└── NumPy zero-copy integration

Month 5-6: DSL Parser Foundation
├── SymPy integration
├── LaTeX parser
├── IR system (PDESystem, PDEEquation)
└── Type database for C++ types
```

**Deliverables:**
- Basic Python API working (Mesh + Field + iteration)
- Equation parser for simple PDEs
- Documentation for first users

### Phase 2: DSL Code Generation (Months 7-12)

**Goal:** End-to-end equation-to-C++ generation

```
Month 7-8: Scheme Generation
├── Flux library implementation
├── Upwind scheme templates
├── Diffusion scheme templates
└── WENO5 scheme templates

Month 9-10: Boundary Conditions
├── BC DSL implementation
├── All BC types (Dirichlet, Neumann, Periodic, Robin)
├── BC inference engine
└── Template generation

Month 11-12: Complete Pipeline
├── Full C++20 generation
├── Time integrators (Euler, RK4, Crank-Nicolson)
├── AMR integration
└── I/O generation
```

**Deliverables:**
- Working DSL for standard PDEs (heat, advection, wave, Burgers)
- Generated code performance within 5% of manual
- Tutorial notebooks

### Phase 3: Scientific ML Integration (Months 13-18)

**Goal:** JAX integration and differentiable solvers

```
Month 13-14: JAX Primitives
├── JAX primitive registration
├── VJP rules for key operations
├── JAX array compatibility
└── Gradient computation

Month 15-16: Differentiable Solvers
├── PINN implementation
├── Neural operator framework
├── Inverse problem examples
└── Optimization loops

Month 17-18: Production Readiness
├── GPU support (CUDA/JAX)
├── MPI integration
├── Performance optimization
└── Complete documentation
```

**Deliverables:**
- Full JAX integration
- Scientific ML examples (PINNs, DeepONet)
- GPU acceleration support
- Release v1.0

---

## 5. Resource Requirements

### 5.1 Team Composition

| Role | FTE | Duration | Responsibilities |
|------|-----|----------|------------------|
| **C++ Architect** | 1.0 | 18 mo | Core design, templates, optimization |
| **Python/DSL Lead** | 1.0 | 18 mo | Python bindings, DSL design, SymPy |
| **Scientific ML Engineer** | 0.5 | 6 mo (mo 13-18) | JAX integration, PINNs |
| **QA/Documentation** | 0.5 | 12 mo | Testing, docs, tutorials |
| **Total** | **3.0 FTE** | - | **18 months** |

### 5.2 Infrastructure

```
Development:
├── CI/CD: GitHub Actions
├── Testing: pytest (Python) + googletest (C++)
├── Documentation: Sphinx + Breathe
├── Benchmarking: Google Benchmark
└── Code coverage: gcov + lcov

Dependencies:
├── Python: ≥3.10
├── pybind11: ≥2.10
├── SymPy: ≥1.12
├── Jinja2: ≥3.1
├── JAX: ≥0.4 (optional)
├── NumPy: ≥1.24
└── xtensor: ≥0.26 (existing)
```

### 5.3 Budget Estimate

| Category | Cost (EUR) | Notes |
|----------|------------|-------|
| Personnel (18 mo) | 300K | 3 FTE × 18 mo × market rate |
| Computing (HPC time) | 20K | Benchmarking, GPU testing |
| Software/licenses | 0K | All OSS |
| Travel/Conferences | 15K | Dissemination |
| **Total** | **335K** | 18-month project |

---

## 6. Risk Management

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **pybind11 compilation issues** | Medium | High | Early prototyping, containerized builds |
| **JAX integration complexity** | High | High | Phase 3 approach, fallback to autograd |
| **Template code bloat** | Medium | Medium | Template instantiation optimization |
| **Performance regression** | Low | High | Continuous benchmarking |
| **SymPy limitations** | Medium | Medium | Custom symbolic operators |

### 6.2 Adoption Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Community resistance** | Low | Medium | Early adopter program, tutorials |
| **Fragmentation (3 APIs)** | Medium | High | Consistent design language |
| **Documentation lag** | Medium | Medium | Docs-first development |
| **Maintenance burden** | High | High | Automated testing, CI |

### 6.3 Strategic Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Competing frameworks** | Medium | Medium | Unique AMR + ML position |
| **Funding interruption** | Low | High | Phased deliverables, modular design |
| **C++ ecosystem evolution** | Low | Low | Standard C++20, minimal deps |

---

## 7. Success Metrics

### 7.1 Technical KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Code reduction** | 100× | LOC per equation (200 → 2) |
| **Performance overhead** | <5% | Benchmark suite |
| **Test coverage** | >80% | gcov/pytest |
| **Documentation completeness** | >90% | Sphinx coverage |
| **Compilation time** | <2 min | CMake profiling |

### 7.2 Adoption KPIs

| Metric | Target (Year 1) | Measurement |
|--------|-----------------|-------------|
| **GitHub stars** | +500 | GitHub analytics |
| **Python downloads** | 1000/month | PyPI stats |
| **Academic citations** | 10 | Google Scholar |
| **Tutorial completions** | 200 | Jupyter notebooks |
| **Community contributors** | 5 | GitHub PRs |

### 7.3 Scientific Impact KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Publications using Samurai** | 5 | Literature survey |
| **ML papers using bindings** | 3 | arXiv search |
| **Teaching adoptions** | 2 | University contacts |
| **Industrial users** | 1 | Case studies |

---

## 8. Governance & Maintenance

### 8.1 Repository Structure

```
samurai/
├── include/samurai/          # C++20 core (existing)
├── python/
│   ├── samurai/              # Python bindings (NEW)
│   │   ├── core/
│   │   ├── schemes/
│   │   ├── algorithms/
│   │   └── io/
│   └── tests/
├── samurai-dsl/
│   ├── samurai_dsl/          # DSL implementation (NEW)
│   │   ├── parser/
│   │   ├── ir/
│   │   ├── codegen/
│   │   └── schemes/
│   ├── examples/
│   └── tests/
├── notebooks/                # Jupyter tutorials (NEW)
│   ├── beginner/
│   ├── intermediate/
│   └── advanced/
└── docs/
    ├── python_api/           # Python docs (NEW)
    ├── dsl_guide/            # DSL guide (NEW)
    └── tutorials/            # Tutorials (NEW)
```

### 8.2 Version Strategy

```
samurai        : C++ core library (e.g., v0.28.0)
samurai-python : Python bindings (e.g., v0.28.0)
samurai-dsl    : DSL package (e.g., v1.0.0)

Release synchronization:
- Major releases: Aligned
- Minor releases: Independent
- Patch releases: Independent
```

### 8.3 Backward Compatibility

```
C++ API:    Stable (SemVer)
Python API: Stable from v1.0 (SemVer)
DSL:        Evolving (May break between major versions)
```

---

## 9. Open Questions & Decisions Needed

### 9.1 Technical Decisions

| Question | Options | Recommendation |
|----------|---------|----------------|
| **Python package manager** | conda, pip, both | Both (conda first, pip later) |
| **JIT compilation** | Numba, ctypes, none | Phase 2 decision |
| **Array ABI** | NumPy only, NumPy+CuPy, agnostic | NumPy first, CuPy later |
| **GPU support** | CUDA, HIP, SYCL | CUDA first (user demand) |
| **Build system** | scikit-build-core, meson, setup.py | scikit-build-core |

### 9.2 Strategic Decisions

| Question | Context | Decision Timeline |
|----------|---------|-------------------|
| **ML framework focus** | JAX vs PyTorch | Month 12 (based on adoption) |
| **Commercial support** | Yes/No/Maybe | Month 18 (post-release) |
| **Foundation governance** | Independent/consortium | Month 12 |

---

## 10. Conclusion

L'intégration du Python Ecosystem (document 05) avec une couche DSL crée une **vision synergique** où:

1. **DSL** sert d'entrée à haut niveau pour les mathématiciens et étudiants
2. **Python API** fournit interactivité et intégration ML
3. **C++20 Core** reste le moteur de performance ultime

**Le succès dépend de:**
- Architecture cohérente avec composants partagés
- Implémentation phasée avec livrables réguliers
- Documentation et exemples de qualité
- Engagement communautaire précoce

**Investissement recommandé:** 335K€, 3 FTE, 18 mois

**ROI attendu:** 10-100× en termes d'adoption, impact scientifique, et contributions communautaires.

---

**Document Version:** 1.0
**Last Updated:** 2025-01-05
**Author:** Samurai V2 Integration Team
**Status:** Ready for Review
