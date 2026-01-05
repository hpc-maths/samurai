# 🥷 Samurai Python Bindings - Documentation Technique

Documentation complète pour le projet de bindings Python de la bibliothèque Samurai AMR/MRA.

## 📚 Structure des Documents

```
md/
├── AGENTS.md                   # 🕵️ Origine, intérêt et liens entre fichiers
├── 00_strategy.md              # Stratégie 8 agents (architecture 3 couches)
├── 01_roadmap.md               # Plan de développement 5 phases (9 mois)
├── 02_technical_feasibility.md # Validation approche technique
├── 03_bindings.md              # Détails implémentation pybind11
├── 04_build_ci.md              # Build system, CMake, CI/CD, wheels
├── 05_ecosystem.md             # Intégration NumPy/SciPy, distribution
├── 06_integrated_roadmap.md    # Vision Python + DSL
├── 07_risk_assessment.md       # 24 risques identifiés + mitigations
├── 08_risk_summary.md          # Version courte des risques
└── 09_risk_dashboard.md        # Indicateurs de surveillance
```

---

## 🎯 Documents par Ordre de Lecture

### **1. Commencer ici** (Vue d'ensemble)

| Fichier | Taille | Description |
|---------|--------|-------------|
| **[00_strategy.md](00_strategy.md)** | 11KB | Stratégie complète - 8 agents analysant les approches de bindings |
| **[01_roadmap.md](01_roadmap.md)** | 15KB | **Document principal** - Roadmap 5 phases, 9 mois, 2.25 FTE |

### **2. Aspects techniques** (Implémentation)

| Fichier | Taille | Description |
|---------|--------|-------------|
| **[02_technical_feasibility.md](02_technical_feasibility.md)** | 39KB | Validation technique - Template instantiation, expression templates |
| **[03_bindings.md](03_bindings.md)** | 46KB | Détails pybind11 - Mesh, Field, Operators, NumPy zero-copy |
| **[04_build_ci.md](04_build_ci.md)** | 45KB | Build system - CMake, scikit-build, CI/CD, PyPI wheels |

### **3. Écosystème & Vision** (Contexte élargi)

| Fichier | Taille | Description |
|---------|--------|-------------|
| **[05_ecosystem.md](05_ecosystem.md)** | 51KB | Intégration Python - NumPy, SciPy, JAX, Jupyter |
| **[06_integrated_roadmap.md](06_integrated_roadmap.md)** | 21KB | Vision Python + DSL synergie |

### **4. Gestion des risques** (Surveillance)

| Fichier | Taille | Description |
|---------|--------|-------------|
| **[07_risk_assessment.md](07_risk_assessment.md)** | 38KB | 24 risques détaillés avec scores et mitigations |
| **[08_risk_summary.md](08_risk_summary.md)** | 11KB | **Version exécutive** - Top 3 risques à surveiller |
| **[09_risk_dashboard.md](09_risk_dashboard.md)** | 9KB | Indicateurs et seuils d'alerte |

---

## 📊 Résumé du Projet

### Architecture 3 Couches

```
┌─────────────────────────────────────────┐
│  Couche 3: Python Convenience Layer     │
│  - API pythonique de haut niveau        │
│  - TimeStepper context managers         │
│  - Visualization Matplotlib             │
└─────────────────────────────────────────┘
                    ↑
┌─────────────────────────────────────────┐
│  Couche 2: Manual Bindings (C++)        │
│  - Operators (diffusion, upwind)        │
│  - AMR adaptation                       │
│  - Zero-copy NumPy integration          │
└─────────────────────────────────────────┘
                    ↑
┌─────────────────────────────────────────┐
│  Couche 1: Generated Bindings (C++)     │
│  - Mesh (1D, 2D, 3D)                    │
│  - ScalarField, VectorField             │
│  - Core algorithms                      │
└─────────────────────────────────────────┘
```

### Roadmap 5 Phases

| Phase | Durée | Objectif | Livrables |
|-------|-------|----------|-----------|
| **1** | 2 mois | Infrastructure & POC | CMake + pybind11, Mesh2D, ScalarField POC |
| **2** | 2 mois | Core API & NumPy | Zero-copy, for_each_cell, VectorField |
| **3** | 2 mois | Operators & Schemes | Diffusion, Upwind, Boundary conditions, AMR |
| **4** | 2 mois | I/O & Testing | HDF5, Test suite >90%, Performance |
| **5** | 1 mois | Python Layer & Distribution | TimeStepper, Documentation, PyPI |

### Budget & Ressources

| Item | Valeur |
|------|--------|
| **Durée** | 9 mois |
| **Équipe** | 2.25 FTE |
| **Budget** | ~200K€ |
| **Confiance** | 78% |

### Top 3 Risques

| Risque | Score | Mitigation |
|--------|-------|------------|
| 🔴 Template instantiation | 9/15 | Type erasure + 20 instantiations |
| 🔴 Memory management | 8.4/15 | pybind11 keep_alive + validation |
| 🟡 Developer resources | 7.5/15 | Financement 2 FTE sécurisé |

---

## 🚀 Pour Commencer

**Nouveau ?** Commencez par lire **[AGENTS.md](AGENTS.md)** pour comprendre l'origine et les liens entre tous les documents.

1. **Pour comprendre la stratégie globale** → Lire `[00_strategy.md](00_strategy.md)`
2. **Pour le plan de développement** → Lire `[01_roadmap.md](01_roadmap.md)`
3. **Pour les détails techniques** → Lire `[02_technical_feasibility.md](02_technical_feasibility.md)` et `[03_bindings.md](03_bindings.md)`
4. **Pour surveiller les risques** → Lire `[08_risk_summary.md](08_risk_summary.md)`

---

## 🔗 Références

- **Repository Samurai**: https://github.com/hpc-maths/samurai
- **Branche pybind11**: `feature/python-bindings`
- **Worktree principal**: `/home/sbstndbs/sbstndbs/samurai-worktrees/main/`
- **Version cible**: 0.28.0-py

---

*Documentation générée par analyse multi-agents avec mode ULTRATHINK*
*Dernière mise à jour: Janvier 2026*
