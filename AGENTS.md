# AGENTS.md — Guide pour les LLM/agents sur samurai

Ce document explique comment un agent (LLM) doit utiliser, compiler, tester et étendre ce projet. Il se concentre sur les chemins de build, les options CMake, les tests, et les démos clés (advection_2d et burgers). Les benchmarks ne sont pas à utiliser (cassés).

## Aperçu du repo
- Code C++ (header-only) dans `include/samurai` (C++20).
- Démos dans `demos/` (surtout `demos/FiniteVolume/`): cibles notables
  - `finite-volume-advection-2d` (MPI prêt)
  - `finite-volume-burgers` (MPI prêt)
- Tests C++ (gtest) et Python (pytest + comparaison HDF5) dans `tests/`.
- CMake racine: options et dépendances; cibles `samurai` (INTERFACE), `tests`, `demos`.
- Environnements Conda/Mamba dans `conda/` (séquentiel et MPI/HDF5 parallèle).
- Workflows CI `.github/workflows/` donnent des recettes de build/exec reproductibles (Linux/macOS, avec/ sans MPI).

## Règles pour agents (esprit et portée)
- Faire des changements ciblés et minimaux; ne pas réorganiser le projet.
- Ne pas utiliser les benchmarks (`-DBUILD_BENCHMARKS=OFF`).
- Conserver le standard C++20 et le modèle header-only (`add_library(samurai INTERFACE)`).
- Pour un nouvel exécutable, lier sur `samurai` (et `CLI11::CLI11` si CLI):
  ```cmake
  add_executable(my_demo main.cpp)
  target_link_libraries(my_demo PRIVATE samurai CLI11::CLI11)
  ```
- Ajouter/adapter des tests si vous modifiez le comportement; favoriser la compilation rapide des tests (voir `SPLIT_TESTS`).

## Dépendances et options CMake utiles
- Options principales (racine `CMakeLists.txt`):
  - `WITH_MPI` (OFF par défaut): build MPI. Requiert HDF5 parallèle (contrôlé via `HDF5_IS_PARALLEL`).
  - `WITH_OPENMP`, `WITH_PETSC` (OFF par défaut).
  - `BUILD_DEMOS`, `BUILD_TESTS`, `SPLIT_TESTS`.
  - `SAMURAI_CHECK_NAN`, `SANITIZERS`, `CLANG_TIDY`, `CPPCHECK`, `IWYU`, `ENABLE_COVERAGE`.
  - Conteneurs: `SAMURAI_FIELD_CONTAINER` (`xtensor`|`eigen3`), `SAMURAI_FLUX_CONTAINER`, `SAMURAI_STATIC_MAT_CONTAINER`.
- Dépendances liées automatiquement: HighFive (HDF5), pugixml, fmt, xtensor/Eigen3, Boost (MPI si `WITH_MPI=ON`).
- PETSc: requis pour certaines démos (e.g. heat/stokes). Les démos advection_2d et burgers n’en ont pas besoin.

## Build rapide (x86) — flags préférés
Nous aimons compiler (surtout les tests) avec: `-march=native -mtune=native -O3 -g`.

- Séquentiel (sans MPI):
  ```bash
  cmake -S . -B build -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DSPLIT_TESTS=ON \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -march=native -mtune=native"
  cmake --build build --parallel
  ```

- MPI (HDF5 parallèle requis):
  ```bash
  mamba env create -f conda/mpi-environment.yml
  mamba activate samurai-env
  cmake -S . -B build -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DWITH_MPI=ON -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DSPLIT_TESTS=ON \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -march=native -mtune=native"
  cmake --build build --parallel
  ```

Conseils
- Ninja (`-GNinja`) accélère les builds multi-cibles.
- Si vous ciblez un seul test/démo, construisez directement sa cible pour réduire le temps.

## Démos clés (exécution locale et MPI)
Les deux démos ci-dessous sont utilisées dans la CI et servent de référence.

### finite-volume-advection-2d
- Cible: `finite-volume-advection-2d` (définie dans `demos/FiniteVolume/CMakeLists.txt`).
- Exécution séquentielle minimale:
  ```bash
  ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05 --nfiles 20
  ```
- Exécution MPI (exemples CI):
  ```bash
  mpiexec -n 4 ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05 --nfiles 20
  ```
- Options utiles (découvertes via le code): `--min-level`, `--max-level`, `--cfl`, `--Ti`, `--Tf`, `--path`, `--filename`, `--nfiles`, `--restart-file`.
- Sorties: fichiers HDF5/xdmf; en MPI, le préfixe inclut `_size_<N>`.

### finite-volume-burgers
- Cible: `finite-volume-burgers`.
- Exécution séquentielle (exemple rapide):
  ```bash
  ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --max-level 8
  ```
- Exécution MPI (exemples CI):
  ```bash
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --max-level 8
  ```
- Options utiles: `--left`, `--right`, `--init-sol` (hat|linear|bands selon dim/n_comp), `--dt`, `--cfl`, `--min-level`, `--max-level`, `--path`, `--filename`, `--nfiles`, `--restart-file`.

## Tests
### Tests C++ (gtest)
- Binaire groupé: `./build/tests/test_samurai_lib`.
- Filtrer un test dans le binaire groupé: `--gtest_filter=SuiteName.TestName`.
- Compilation plus rapide par test: activer `-DSPLIT_TESTS=ON` puis construire/exec un exécutable par fichier, ex.:
  ```bash
  cmake --build build --target test_field
  ./build/tests/test_field
  ```

### Tests Python (pytest + comparaison HDF5)
- Les tests Python lancent des exécutables de `build/demos/...` et comparent les fichiers HDF5 générés aux références (`tests/reference/...`).
- Avant `pytest`, assurez-vous d’avoir construit les démos visées.
- Exécution typique:
  ```bash
  cd tests
  pytest -v -s --h5diff
  ```
- Générer/mettre à jour des références HDF5:
  ```bash
  cd tests
  pytest -k <pattern> --h5diff --h5diff-generate-ref
  ```

## Bonnes pratiques de modification
- Respecter le style existant; ne pas introduire de dépendances non nécessaires.
- Pour une nouvelle démo, suivre les modèles des fichiers `demos/FiniteVolume/*.cpp` et lier `samurai` + `CLI11::CLI11`.
- Pour un nouveau test C++, créer `tests/<nom>.cpp` et l’ajouter à la liste `SAMURAI_TESTS` (voir `tests/CMakeLists.txt`). Avec `SPLIT_TESTS=ON`, un exécutable par fichier est généré automatiquement.
- Pour vérifier le parallélisme MPI, réutiliser la structure de la CI (`.github/workflows/ci.yml`) et les scripts `python/compare.py`.

## Conventions PR & commits
- Conventions de message: types `perf:`, `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`, `style:`, `revert:` (scope optionnel), format `type(scope?): sujet`.
- Titres très synthétiques; description brève et factuelle. N’étoffer que pour une très grosse feature (rare).
- Toujours exécuter `pre-commit run --all-files` avant push/PR pour standardiser le code et les fichiers.
- PR: rester concis, lier l’issue si pertinent, vérifier que la CI passe.

## Exécution rapide des démos (dev local)
- Pour itérer vite: utiliser `--Tf 0.1` voire `--Tf 0.01`, `--max-level 8/9`, `--min-level 5`.
- Exemples:
  - Advection 2D (séquentiel):
    ```bash
    ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.1 --nfiles 20 --min-level 5 --max-level 8
    ```
  - Burgers (séquentiel):
    ```bash
    ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --min-level 5 --max-level 8
    ```
- En MPI, garder de petits `--Tf` et niveaux similaires pour limiter le temps/IO.

## Compilation: prudence mémoire/temps
- La compilation peut être lente et gourmande en mémoire: construire des cibles spécifiques plutôt que `all`.
- Utiliser Ninja (`-GNinja`) et limiter le parallélisme si besoin (`cmake --build build --parallel <N>`).
- Activer `-DSPLIT_TESTS=ON` et ne construire/faire tourner que les tests nécessaires.
- Éviter les reconstructions inutiles: modifier peu de fichiers, et préférer des changements localisés.

## Options de debug et qualité
- `-DSAMURAI_CHECK_NAN=ON` pour détecter les NaN pendant les calculs.
- `-DSANITIZERS=ON` pour ASan/UBSan en phase de test (éviter en perf).
- Intégrations qualité: `CLANG_TIDY`, `CPPCHECK`, `IWYU`, `ENABLE_COVERAGE` (voir CI pour exemples d’usage).

## Limitations et notes
- Benchmarks: cassés, ne pas utiliser (`-DBUILD_BENCHMARKS=OFF`).
- MPI: exige une build HDF5 parallèle; CMake échouera explicitement sinon.
- PETSc: nécessaire uniquement pour certaines démos; inutile pour advection_2d/burgers.

## Exemples synthétiques (copiés de la CI)
- Build + tests (Linux, séquentiel, PETSc ON pour d’autres démos):
  ```bash
  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DWITH_PETSC=ON -DBUILD_DEMOS=ON -DBUILD_TESTS=ON
  cmake --build build --target all --parallel 2
  ./build/tests/test_samurai_lib
  (cd tests && pytest -v -s --h5diff)
  ```
- Build + exécution MPI (advection_2d & burgers):
  ```bash
  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=ON -DBUILD_DEMOS=ON -DBUILD_TESTS=ON
  cmake --build build --target finite-volume-advection-2d --parallel 4
  cmake --build build --target finite-volume-burgers --parallel 4
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05 --nfiles 20
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --max-level 8
  ```
