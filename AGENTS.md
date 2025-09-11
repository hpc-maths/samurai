# AGENTS.md — Guide for LLM/agents on samurai

Concise rules for using, building, testing, and extending this project with AI coding assistants. Benchmarks are broken — do not use them.

## Why This File Matters
- Many AI coding assistants and IDEs automatically read AGENTS.md to adapt to a repository’s rules (e.g., Codex CLI, Cursor, Google Gemini, Warp AI, etc.).
- This file exists to help users working with LLMs avoid unsafe or wasteful changes and to follow project-specific conventions (build paths, CMake flags, tests, demo settings).
- As agent usage grows, AGENTS.md ensures agents “do the right thing” by default: make small, safe, targeted edits aligned with this repository.
- Keep this file up to date; its scope is the whole repository unless otherwise specified.

## Repository Overview
- Header-only C++20 library in `include/samurai` (target `samurai`).
- Demos in `demos/FiniteVolume`: `finite-volume-advection-2d`, `finite-volume-burgers` (MPI-ready).
- Tests: C++ (gtest) and Python (pytest + HDF5) in `tests/`.
- Environments: `conda/` (sequential and MPI/HDF5 parallel).
- CI workflows: `.github/workflows/` (reference build/run recipes).

## Core Rules (scope & spirit)
- Prefer small, targeted changes; do not reorganize the project.
- Do not use benchmarks (`-DBUILD_BENCHMARKS=OFF`).
- Keep C++20 + header-only model (`add_library(samurai INTERFACE)`).
- New executable links: `samurai` (+ `CLI11::CLI11` if CLI).
- Update/add tests when changing behavior; enable `SPLIT_TESTS` for faster builds.

## Communication & Language
- All commit messages, PR titles/descriptions, code comments, and documentation must be in English only.

## Key CMake Options (see root CMakeLists.txt for full list)
- `WITH_MPI` (OFF): MPI build; requires parallel HDF5 (`HDF5_IS_PARALLEL`).
- `BUILD_DEMOS`, `BUILD_TESTS`, `SPLIT_TESTS` (faster per-test targets).
- Debug/quality: `SAMURAI_CHECK_NAN`, `SANITIZERS`, `CLANG_TIDY`, `CPPCHECK`, `IWYU`, `ENABLE_COVERAGE`.
- PETSc needed only for some demos (not for advection_2d/burgers).

## Build Quickstart (x86)
- Preferred flags (esp. tests): `-march=native -mtune=native -O3 -g`.
- Sequential:
  ```bash
  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DSPLIT_TESTS=ON \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -march=native -mtune=native"
  cmake --build build --parallel
  ```
- MPI (parallel HDF5 required):
  ```bash
  mamba env create -f conda/mpi-environment.yml
  mamba activate samurai-env
  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DWITH_MPI=ON -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DSPLIT_TESTS=ON \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -march=native -mtune=native"
  cmake --build build --parallel
  ```
- Tips: use Ninja; build specific targets; cap parallelism if memory is tight.

## Demos (local and MPI)
- Targets: `finite-volume-advection-2d`, `finite-volume-burgers` (see `demos/FiniteVolume/`).
- Quick local runs for iteration:
  ```bash
  ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.1 --nfiles 20 --min-level 5 --max-level 8
  ./build/demos/FiniteVolume/finite-volume-burgers        --Tf 0.1 --nfiles 20 --min-level 5 --max-level 8
  ```
- MPI examples (adjust `-n`):
  ```bash
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05 --nfiles 20
  mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-burgers        --Tf 0.1  --nfiles 20 --max-level 8
  ```
- Typical useful flags: `--min-level`, `--max-level`, `--cfl`, `--Ti`, `--Tf`, `--path`, `--filename`, `--nfiles`, `--restart-file` (check `--help`).
- Outputs: HDF5/XDMF; MPI filenames include `_size_<N>`.

## Tests
- C++ (gtest): aggregated binary `./build/tests/test_samurai_lib`; filter with `--gtest_filter=Suite.Test`. With `-DSPLIT_TESTS=ON`, build and run a single test target (e.g., `test_field`).
- Python (pytest + HDF5): builds demos in `build/demos/...`, compares against `tests/reference/...`.
  ```bash
  (cd tests && pytest -v -s --h5diff)
  (cd tests && pytest -k <pattern> --h5diff --h5diff-generate-ref)
  ```

## Modification Best Practices
- Follow existing style; avoid unnecessary dependencies.
- New demo: follow `demos/FiniteVolume/*.cpp`; link `samurai` (+ `CLI11::CLI11` if CLI).
- New C++ test: add `tests/<name>.cpp` to `SAMURAI_TESTS`; with `SPLIT_TESTS=ON`, one exe per file.
- MPI checks: mirror CI (`.github/workflows/ci.yml`) and `python/compare.py`.

## PR & Commit Conventions
- Types: `perf:`, `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`, `style:`, `revert:`; format `type(scope?): subject`.
- Titles very concise; body brief and factual. Elaborate only for large features.
- Run `pre-commit run --all-files` before push/PR. Keep PRs small; ensure CI passes; link issues.
- English only for commits, PRs, comments, and docs.

## Fast Demo Tips
- Iterate with small `--Tf` (0.1 or 0.01), `--min-level 5`, `--max-level 8/9`.
- In MPI, keep `--Tf` small and levels similar to reduce runtime/IO.

## Build Performance
- Build only needed targets; avoid `all`. Use Ninja and limit `--parallel` if RAM is constrained.
- Enable `-DSPLIT_TESTS=ON`; build/run only the tests you need.
- Keep edits localized to minimize rebuilds.

## Debug and Quality
- `-DSAMURAI_CHECK_NAN=ON`, `-DSANITIZERS=ON` (testing only), `CLANG_TIDY`, `CPPCHECK`, `IWYU`, `ENABLE_COVERAGE`.

## Limitations
- Benchmarks broken (use `-DBUILD_BENCHMARKS=OFF`).
- MPI requires parallel HDF5; CMake fails explicitly otherwise.
- PETSc only for some demos; not needed for advection_2d/burgers.
## References
- See `CMakeLists.txt` for all options and `.github/workflows/` for CI recipes.
