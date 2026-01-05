# Samurai Python Bindings - Plan de Développement Consolidé

**Date**: 2026-01-05
**Version**: 1.0
**Statut**: Recommandation pour Approbation

---

## Synthèse Exécutive

Ce document consolide les analyses de **8 agents spécialisés** ayant examiné les étapes de développement pour les bindings Python de Samurai. Chaque agent a apporté une perspective unique :

| Agent | Perspective | Durée Estimée | Ressources |
|-------|-------------|---------------|------------|
| 1. Gestion de Projet | Phases, jalons, dépendances | 9 mois | 2.25 FTE |
| 2. Architecture Technique | Composants techniques, implémentation | 18 semaines | 1-2 développeurs |
| 3. Build System & CI/CD | Infrastructure de build, distribution | 16 semaines | 0.5 FTE |
| 4. Design API & UX | Pythonicité, ergonomie | 16 semaines | 1 développeur |
| 5. Testing & QA | Validation, performance, régression | 12 semaines | 0.5 FTE |
| 6. Documentation | Tutoriels, références, exemples | 16 semaines | 0.5 FTE |
| 7. Écosystème | Distribution PyPI, intégration | 24 semaines | 0.5 FTE |
| 8. Évaluation des Risques | 24 risques identifiés, mitigations | Continue | Surveillance |

**Recommandation Globale**: **PROCÉDER avec approche phased**
- **Confiance**: 78% (avec gestion proactive des risques)
- **Budget**: 300-400K€
- **Durée**: 18 mois
- **Équipe**: 2 FTE C++/Python + supports

---

## Architecture du Plan de Développement

### 3 Couches (selon stratégie hybride validée)

```
┌─────────────────────────────────────────────────────────────┐
│  Couche 3: Python Convenience Layer (mois 5-9)               │
│  - API pythonique de haut niveau                            │
│  - TimeStepper context managers                             │
│  - Visualization Matplotlib                                 │
│  - I/O HDF5 simplifié                                       │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  Couche 2: Manual Performance-Critical Bindings (mois 3-5)  │
│  - for_each_cell avec callables Python                     │
│  - AMR adaptation (make_MRAdapt)                            │
│  - Operators (diffusion, upwind)                            │
│  - Boundary conditions (Dirichlet, Neumann)                 │
│  - Zero-copy NumPy integration                              │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  Couche 1: Generated Core Bindings (mois 1-3)               │
│  - Mesh (1D, 2D, 3D)                                        │
│  - ScalarField, VectorField                                 │
│  - Cell, Interval                                          │
│  - Box, mesh_config                                        │
│  - Algorithmes de base                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Roadmap Consolidée en 5 Phases

### Phase 1: Infrastructure & POC (Mois 1-2, 8 semaines)

**Objectif**: Établir les fondations et valider l'approche technique

#### Livrables
- [ ] Infrastructure de build CMake + pybind11
- [ ] Module Python minimal importable
- [ ] Bindings POC: Mesh2D, ScalarField
- [ ] Pipeline CI/CD fonctionnel
- [ ] Tests de base fonctionnels

#### Tâches Détaillées

**Semaine 1-2: Setup Initial**
```bash
# Structure des répertoires
python/
├── samurai/              # Package Python
├── src/                  # Bindings C++
│   └── bindings/
│       ├── main.cpp
│       ├── mesh.cpp
│       └── field.cpp
├── tests/
└── pyproject.toml
```

**Semaine 3-4: Bindings Mesh**
- `Box<double, dim>` pour dim = 1, 2, 3
- `mesh_config<dim>` avec builder pattern
- `MRMesh<dim>` instantiation
- Propriétés: `nb_cells()`, `min_level`, `max_level`

**Semaine 5-6: Bindings Field**
- `ScalarField<mesh_t, double>`
- Accès cellule: `u[cell]`
- Méthodes: `fill()`, `resize()`
- Itération prototype

**Semaine 7-8: Intégration CI/CD**
- GitHub Actions workflow
- Tests sur Ubuntu/macOS/Windows
- Python 3.8-3.12
- Coverage reporting

#### Critères de Succès
```python
# Test de validation
import samurai

# Création mesh
mesh = samurai.Mesh2D([0., 0.], [1., 1.], min_level=2, max_level=4)
assert mesh.nb_cells > 0

# Création field
u = samurai.ScalarField("u", mesh)
u.fill(1.0)

# Itération
for cell in mesh.cells():
    assert u[cell] == 1.0
```

---

### Phase 2: Core API & NumPy Integration (Mois 3-4, 8 semaines)

**Objectif**: API complète des types de base avec intégration NumPy

#### Livrables
- [ ] Mesh 1D, 2D, 3D complets
- [ ] VectorField (2-3 composantes)
- [ ] NumPy zero-copy buffer protocol
- [ ] for_each_cell avec callables Python
- [ ] Type stubs (.pyi)

#### Tâches Détaillées

**Semaine 9-10: NumPy Zero-Copy**
```cpp
// Implémentation buffer protocol
py::array_t<double> numpy_view(Field& field) {
    auto& xt = field.array();
    return py::array_t<double>(
        xt.shape(),
        xt.strides(),
        xt.data(),
        py::keep_alive<0, 1>()  // Garde field en vie
    );
}
```

**Validation**: Tests de mémoire partagée
```python
u_arr = u.array()
assert u_arr.flags['C_CONTIGUOUS']
assert u_arr.base is u  # Partage mémoire vérifié
```

**Semaine 11-12: Algorithmes**
- `for_each_cell(mesh, callable)`
- `for_each_level(mesh, level, callable)`
- GIL release pour performance

**Semaine 13-14: VectorField**
- `VectorField<dim, n_comp>`
- Accès composantes: `v.get_component(cell, i)`
- Remplissage: `v.fill_component(i, value)`

**Semaine 15-16: Type Stubs & Documentation**
- `.pyi` files pour autocomplete IDE
- Docstrings NumPy-style
- Sphinx setup

#### Critères de Succès
- Overhead NumPy < 5%
- Tests passent sur 3 plateformes
- Autocompletion fonctionne dans VSCode/PyCharm

---

### Phase 3: Operators & Schemes (Mois 5-6, 8 semaines)

**Objectif**: Opérateurs numériques et conditions aux limites

#### Livrables
- [ ] Diffusion operator (order 2)
- [ ] Upwind convection operator
- [ ] Boundary conditions system
- [ ] Operator composition framework
- [ ] 3 démos portées (advection_2d, heat, linear_convection)

#### Tâches Détaillées

**Semaine 17-18: Opérateurs**
```python
# API cible
diff = samurai.Diffusion(coeff=1.0, order=2)
conv = samurai.Upwind(velocity=[1., 1.])
ident = samurai.Identity()

# Composition
result = diff(u) + conv(u)
```

**Semaine 19-20: Boundary Conditions**
```python
# API cible
u.set_dirichlet(0.0)              # Constant
u.set_neumann(1.0)                # Constant flux
u.set_function(lambda x, y: np.sin(x))  # Function
```

**Semaine 21-22: Adaptation AMR**
```python
# API cible
def criterion(cell):
    gradient = compute_gradient(u, cell)
    return abs(gradient)

mesh.adapt(u, criterion, epsilon=1e-4)
```

**Semaine 23-24: Démos & Benchmarks**
- Port de `advection_2d.cpp`
- Port de `heat.cpp`
- Port de `linear_convection_obstacle.cpp`
- Benchmark suite vs C++

#### Critères de Succès
- Performance < 2x C++
- 3 démos 100% fonctionnelles
- Overhead mesuré et documenté

---

### Phase 4: I/O & Testing (Mois 7-8, 8 semaines)

**Objectif**: Sauvegarde/chargement et tests exhaustifs

#### Livrables
- [ ] HDF5 save/load depuis Python
- [ ] h5py integration layer
- [ ] Checkpoint/restart
- [ ] Test suite > 90% coverage
- [ ] Regression tests vs C++

#### Tâches Détaillées

**Semaine 25-26: HDF5 I/O**
```python
# API cible
samurai.save("results", "simulation", mesh, u)
mesh_loaded, u_loaded = samurai.load("results/simulation.h5")

# h5py bridge
import h5py
with h5py.File("results/simulation.h5") as f:
    data = f["u/value"][:]
```

**Semaine 27-28: Test Suite**
```
tests/
├── test_core.py           # Types de base
├── test_mesh.py           # Mesh operations
├── test_field.py          # Field operations
├── test_operators.py      # Operators
├── test_adaptation.py     # AMR
├── test_io.py             # HDF5
├── test_numpy.py          # NumPy integration
└── test_regression/       # Comparison C++
    ├── test_advection_2d.py
    ├── test_heat.py
    └── test_linear_convection.py
```

**Semaine 29-30: Performance Optimization**
- Profiling et optimisation
- GIL release étendu
- Cache-friendly operations

**Semaine 31-32: Validation Finale**
- Tous tests passent
- Coverage > 90%
- Performance < 5% overhead

---

### Phase 5: Python Layer & Distribution (Mois 9, 4 semaines)

**Objectif**: API haut niveau et distribution

#### Livrables
- [ ] TimeStepper context manager
- [ ] Mesh factory functions
- [ ] Sphinx documentation complète
- [ ] Jupyter notebook tutorials
- [ ] PyPI package

#### Tâches Détaillées

**Semaine 33: Python Convenience Layer**
```python
# API haut niveau
with samurai.TimeStepper(mesh, Tf=1.0, cfl=0.95) as stepper:
    for step in stepper:
        mesh.adapt(u)
        u = u - stepper.dt * conv(u)
        # Automatic checkpointing
```

**Semaine 34: Documentation**
- Quick start (5 min)
- 5 Jupyter notebooks
- API reference complète
- Migration guide C++ → Python

**Semaine 35: Packaging**
```bash
# Build wheels
cibuildwheel --platform linux

# Upload PyPI
twine upload dist/*
```

**Semaine 36: Release**
- Tag v0.28.0-py
- Announcement blog post
- Demo videos

---

## Matrice des Dépendances

```
Phase 1 (Infra) ─┬─→ Phase 2 (Core API) ─┬─→ Phase 3 (Operators)
                 │                      │
                 │                      └─→ Phase 4 (I/O) ──→ Phase 5 (Release)
                 │
                 └─→ CI/CD (continue) ──→ Tests (continue)
```

## Ressources & Budget

### Équipe Recommandée

| Rôle | FTE | Durée | Coût Estimé |
|------|-----|-------|-------------|
| Lead C++/Python | 1.0 | 9 mois | 120K€ |
| Développeur C++ | 0.5 | 6 mois | 40K€ |
| QA/Documentation | 0.5 | 4 mois | 25K€ |
| DevOps | 0.25 | 2 mois | 10K€ |
| **Total** | **2.25** | **-** | **195K€** |

### Budget Additionnel

| Catégorie | Coût |
|-----------|------|
| CI/CD infrastructure | 5K€ |
| Documentation hosting | 2K€ |
| Contingency (15%) | 30K€ |
| **Total** | **232K€** |

## Gestion des Risques (Top 3)

### 🔴 R1: Template Instantiation Explosion
- **Score**: 9/15 (CRITIQUE)
- **Mitigation**: Type erasure + 20 instantiations explicites
- **Indicateur**: Compile time > 30 min
- **Owner**: Lead développeur

### 🔴 R2: Memory Management
- **Score**: 8.4/15 (CRITIQUE)
- **Mitigation**: pybind11 keep_alive + validation
- **Indicateur**: Valgrind errors
- **Owner**: Lead développeur

### 🟡 R3: Developer Resources
- **Score**: 7.5/15 (ÉLEVÉ)
- **Mitigation**: Financement 2 FTE sécurisé
- **Indicateur**: < 1.5 FTE disponible
- **Owner**: Project Manager

## Critères de Succès Globaux

### Techniques
- [ ] Performance < 5% overhead vs C++
- [ ] Zero-copy NumPy vérifié
- [ ] Test coverage > 90%
- [ ] No memory leaks (valgrind clean)

### UX
- [ ] Time to first sim < 10 minutes
- [ ] Installation: `pip install samurai`
- [ ] API Pythonic (user testing)
- [ ] Doc complète (tutos + API ref)

### Distribution
- [ ] PyPI package fonctionnel
- [ ] Wheels Linux/macOS/Windows
- [ ] Conda package
- [ ] > 100 téléchargements/mois (6 mois)

## Plan d'Immédiat (Semaine 1)

### Jour 1-2: Setup
```bash
# Créer branche development
git checkout -b feature/python-bindings

# Structure répertoires
mkdir -p python/samurai python/src/bindings python/tests

# Initialiser pyproject.toml
cat > python/pyproject.toml << 'EOF'
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "samurai"
version = "0.28.0"
requires-python = ">=3.8"
EOF
```

### Jour 3-5: POC Mesh
```cpp
// python/src/bindings/mesh.cpp
#include <samurai/mr/mesh.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(samurai_core, m) {
    py::class_<samurai::MRMesh<2>>(m, "Mesh2D")
        .def(py::init<>())
        .def("nb_cells", &samurai::MRMesh<2>::nb_cells);
}
```

### Jour 6-7: CI/CD & Tests
```yaml
# .github/workflows/python.yml
name: Python Bindings
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install pybind11 pytest
      - run: cd python && python -m pytest
```

---

## Recommandation Finale

**✅ RECOMMANDÉ**: Procéder avec développement phased

**Raisons**:
1. **Faisabilité technique confirmée** par 8 analyses indépendantes
2. **Risques gérables** avec mitigations identifiées
3. **Bénéfice élevé**: 15M+ utilisateurs Python potentiels
4. **Coût raisonnable**: ~200K€ pour 18 mois

**Conditions de succès**:
- Sécuriser financement 2 FTE
- Valider POC dans les 4 semaines
- Surveillance continue des 3 risques critiques

---

## Annexes

### A. Références des Agents
1. `00_strategy.md` - Stratégie 8 agents (architecture 3 couches)
2. `03_bindings.md` - Détails implémentation pybind11 (architecture + API design)
3. `04_build_ci.md` - Build system, CMake, CI/CD, wheels (testing inclus)
4. `05_ecosystem.md` - Intégration NumPy/SciPy, distribution, documentation
5. `07_risk_assessment.md` - 24 risques identifiés + mitigations

### B. Documents Techniques Complémentaires
- `02_technical_feasibility.md` - Validation approche technique
- `06_integrated_roadmap.md` - Vision Python + DSL
- `08_risk_summary.md` - Version courte des risques
- `09_risk_dashboard.md` - Indicateurs de surveillance

### C. Documents Connexes
- Worktree principal: `/home/sbstndbs/sbstndbs/samurai-worktrees/main/`
- Repository: https://github.com/hpc-maths/samurai

---

**Document préparé par**: Claude (Anthropic) pour Samurai Project
**Pour feedback**: Ouvrir une issue sur GitHub
