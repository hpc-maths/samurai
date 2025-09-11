# AGENTS.md — Guide for LLM/agents on samurai

This document explains how an agent (LLM) should use, build, test, and extend this project. It focuses on build paths, CMake options, tests, and the key demos (advection_2d and burgers). Do not use benchmarks (they are broken).

## Repository Overview
- Header-only C++ code in `include/samurai` (C++20).
- Demos in `demos/` (mainly `demos/FiniteVolume/`): notable targets
  - `finite-volume-advection-2d` (MPI-ready)
  - `finite-volume-burgers` (MPI-ready)
- C++ tests (gtest) and Python tests (pytest + HDF5 comparison) in `tests/`.
- Root CMake: options and dependencies; targets `samurai` (INTERFACE), `tests`, `demos`.
- Conda/Mamba environments in `conda/` (sequential and parallel HDF5/MPI).
- CI workflows in `.github/workflows/` provide reproducible build/exec recipes (Linux/macOS, with/without MPI).

## Rules for Agents (scope & spirit)
- Make minimal, targeted changes; do not reorganize the project.
- Do not use the benchmarks (`-DBUILD_BENCHMARKS=OFF`).
- Keep C++20 and the header-only model (`add_library(samurai INTERFACE)`).
- For a new executable, link against `samurai` (and `CLI11::CLI11` if it has a CLI):
  ```cmake
  add_executable(my_demo main.cpp)
  target_link_libraries(my_demo PRIVATE samurai CLI11::CLI11)
  ```
- Add/update tests when changing behavior; prefer fast test compilation (see `SPLIT_TESTS`).

## Communication & Language
- All commit messages, PR titles/descriptions, code comments, and documentation must be in English only.

## Useful Dependencies and CMake Options
- Main options (root `CMakeLists.txt`):
  - `WITH_MPI` (OFF by default): build with MPI. Requires parallel HDF5 (controlled via `HDF5_IS_PARALLEL`).
  - `WITH_OPENMP`, `WITH_PETSC` (OFF by default).
  - `BUILD_DEMOS`, `BUILD_TESTS`, `SPLIT_TESTS`.
  - `SAMURAI_CHECK_NAN`, `SANITIZERS`, `CLANG_TIDY`, `CPPCHECK`, `IWYU`, `ENABLE_COVERAGE`.
  - Containers: `SAMURAI_FIELD_CONTAINER` (`xtensor`|`eigen3`), `SAMURAI_FLUX_CONTAINER`, `SAMURAI_STATIC_MAT_CONTAINER`.
- Linked automatically as needed: HighFive (HDF5), pugixml, fmt, xtensor/Eigen3, Boost (MPI when `WITH_MPI=ON`).
- PETSc: required for some demos (e.g., heat/stokes). Not needed for advection_2d and burgers.

## Fast Build (x86) — preferred flags
We prefer building (especially tests) with: `-march=native -mtune=native -O3 -g`.

- Sequential (no MPI):
  ```bash
  cmake -S . -B build -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DSPLIT_TESTS=ON \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -march=native -mtune=native"
  cmake --build build --parallel
  ```

- MPI (requires parallel HDF5):
  ```bash
  mamba env create -f conda/mpi-environment.yml
  mamba activate samurai-env
  cmake -S . -B build -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DWITH_MPI=ON -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DSPLIT_TESTS=ON \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -march=native -mtune=native"
  cmake --build build --parallel
  ```

Tips
- Ninja (`-GNinja`) speeds up multi-target builds.
- If you only need one test/demo, build that target directly to save time.

## Key Demos (local and MPI runs)
The two demos below are used in CI and serve as references.

### finite-volume-advection-2d
- Target: `finite-volume-advection-2d` (defined in `demos/FiniteVolume/CMakeLists.txt`).
- Minimal sequential run:
  ```bash
  ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05 --nfiles 20
  ```
- MPI run (CI examples):
  ```bash
  mpiexec -n 4 ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05 --nfiles 20
  ```
- Useful options (discoverable in code): `--min-level`, `--max-level`, `--cfl`, `--Ti`, `--Tf`, `--path`, `--filename`, `--nfiles`, `--restart-file`.
- Outputs: HDF5/XDMF files; in MPI, filenames include `_size_<N>`.

### finite-volume-burgers
- Target: `finite-volume-burgers`.
- Sequential (quick example):
  ```bash
  ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --max-level 8
  ```
- MPI run (CI examples):
  ```bash
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --max-level 8
  ```
- Useful options: `--left`, `--right`, `--init-sol` (hat|linear|bands depending on dim/n_comp), `--dt`, `--cfl`, `--min-level`, `--max-level`, `--path`, `--filename`, `--nfiles`, `--restart-file`.

## Tests
### C++ tests (gtest)
- Aggregated binary: `./build/tests/test_samurai_lib`.
- Filter a test in the aggregated binary: `--gtest_filter=SuiteName.TestName`.
- Faster per-test builds: enable `-DSPLIT_TESTS=ON`, then build/run the single test executable, e.g.:
  ```bash
  cmake --build build --target test_field
  ./build/tests/test_field
  ```

### Python tests (pytest + HDF5 comparison)
- Python tests run executables from `build/demos/...` and compare generated HDF5 files to references (`tests/reference/...`).
- Before `pytest`, make sure the relevant demos are built.
- Typical run:
  ```bash
  cd tests
  pytest -v -s --h5diff
  ```
- Generate/update HDF5 references:
  ```bash
  cd tests
  pytest -k <pattern> --h5diff --h5diff-generate-ref
  ```

## Modification Best Practices
- Respect the existing style; do not add unnecessary dependencies.
- For a new demo, follow the patterns in `demos/FiniteVolume/*.cpp` and link against `samurai` + `CLI11::CLI11`.
- For a new C++ test, create `tests/<name>.cpp` and add it to the `SAMURAI_TESTS` list (see `tests/CMakeLists.txt`). With `SPLIT_TESTS=ON`, one executable per file is generated automatically.
- To verify MPI behavior, reuse the CI structure (`.github/workflows/ci.yml`) and the `python/compare.py` scripts.

## PR & Commit Conventions
- Message conventions: types `perf:`, `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`, `style:`, `revert:` (optional scope), format `type(scope?): subject`.
- Titles must be very concise; keep the body short and factual. Only add detail for truly large features (rare).
- Always run `pre-commit run --all-files` before push/PR to standardize code and files.
- PRs: keep them concise, link the relevant issue if applicable, and ensure CI passes.
- Language: English only for commits, PRs, comments, and docs.

## Fast Demo Runs (local dev)
- For quick iterations: use `--Tf 0.1` or even `--Tf 0.01`, `--max-level 8/9`, `--min-level 5`.
- Examples:
  - Advection 2D (sequential):
    ```bash
    ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.1 --nfiles 20 --min-level 5 --max-level 8
    ```
  - Burgers (sequential):
    ```bash
    ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --min-level 5 --max-level 8
    ```
- In MPI, keep small `--Tf` and similar levels to limit runtime and IO.

## Compilation: memory/time caution
- Builds can be slow and memory-hungry: build specific targets instead of `all`.
- Use Ninja (`-GNinja`) and limit parallelism if needed (`cmake --build build --parallel <N>`).
- Enable `-DSPLIT_TESTS=ON` and only build/run the tests you need.
- Avoid unnecessary rebuilds: touch as few files as possible and keep changes localized.

## Debug and Quality Options
- `-DSAMURAI_CHECK_NAN=ON` to detect NaNs during computations.
- `-DSANITIZERS=ON` for ASan/UBSan in testing (avoid for performance runs).
- Quality integrations: `CLANG_TIDY`, `CPPCHECK`, `IWYU`, `ENABLE_COVERAGE` (see CI for examples).

## Limitations and Notes
- Benchmarks: broken, do not use (`-DBUILD_BENCHMARKS=OFF`).
- MPI: requires a parallel HDF5 build; CMake will fail explicitly otherwise.
- PETSc: only needed for some demos; not needed for advection_2d/burgers.

## Synthetic Examples (from CI)
- Build + tests (Linux, sequential, PETSc ON for other demos):
  ```bash
  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DWITH_PETSC=ON -DBUILD_DEMOS=ON -DBUILD_TESTS=ON
  cmake --build build --target all --parallel 2
  ./build/tests/test_samurai_lib
  (cd tests && pytest -v -s --h5diff)
  ```
- Build + MPI runs (advection_2d & burgers):
  ```bash
  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=ON -DBUILD_DEMOS=ON -DBUILD_TESTS=ON
  cmake --build build --target finite-volume-advection-2d --parallel 4
  cmake --build build --target finite-volume-burgers --parallel 4
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05 --nfiles 20
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-burgers --Tf 0.1 --nfiles 20 --max-level 8
  ```
