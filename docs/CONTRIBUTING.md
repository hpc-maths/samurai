# Contributing to samurai

Thank you for your interest in samurai.
Bug reports, questions, documentation fixes and code all improve the library, and each of them is welcome.
Everyone who takes part follows the [code of conduct](CODE_OF_CONDUCT.md).

This guide takes you from a fresh clone to a passing test suite, and from a typo in the documentation to a merged pull request.

## Ways to contribute

- **Report a bug**: open an issue with the [bug report form](https://github.com/hpc-maths/samurai/issues/new?template=bug_report.yml).
  Search the [existing issues](https://github.com/hpc-maths/samurai/issues?q=is%3Aissue) first: the problem may already be known or fixed.
- **Request a feature**: open an issue with the [feature request form](https://github.com/hpc-maths/samurai/issues/new?template=new_features.yml) and describe what you want to do with samurai.
- **Ask a question**: use the [Q&A category of GitHub Discussions](https://github.com/hpc-maths/samurai/discussions/categories/q-a).
- **Fix the documentation**: every page of the [documentation](https://hpc-math-samurai.readthedocs.io) is a Markdown file under `docs/source/`.
  A typo fix needs no local build: edit the file on GitHub and open a pull request.
  See [Build the documentation](#build-the-documentation) for larger changes.
- **Contribute code**: for a large change, open an issue or a discussion first so that the design can be agreed on before you write it.

## Set up a development environment

### Prerequisites

- Git, and a [conda-forge](https://conda-forge.org/download/) installation: `conda`, `mamba` or `micromamba`.
  The commands below use `conda`; replace it with `mamba` or `micromamba` if you use one of those.
- A C++20 compiler.
  The CI builds samurai with GCC 11 to 13 and Clang 16 to 19 on Linux, and with the conda-forge compiler on macOS.

### Get the sources and the dependencies

1. Fork [hpc-maths/samurai](https://github.com/hpc-maths/samurai) on GitHub, then clone your fork:

   ```bash
   git clone https://github.com/<your-user>/samurai.git
   cd samurai
   ```

2. Create the `samurai-env` environment from `conda/environment.yml`.
   It provides CMake, Ninja, xtensor, HighFive, fmt, pugixml, CLI11 and the Python packages of the test suite:

   ```bash
   conda env create -f conda/environment.yml
   conda activate samurai-env
   ```

3. Install a compiler and pre-commit in the same environment.
   `cxx-compiler` is the conda-forge compiler; skip it if you use a system compiler.

   ```bash
   conda install -c conda-forge cxx-compiler pre-commit
   ```

   On Linux, if CMake later stops with `CMAKE_AR-NOTFOUND`, your system has no `ar`: install it with `conda install -c conda-forge binutils`.

4. Install the pre-commit hooks so that they run on each `git commit`:

   ```bash
   pre-commit install
   ```

### Configure with a CMake preset

`CMakePresets.json` defines four configure presets.
Each one uses Ninja, writes `compile_commands.json`, enables ccache or sccache when one is installed, and builds in `build/<preset>`.

| Preset | Build type | Use it for |
| --- | --- | --- |
| `dev` | Debug (`-O0 -g`) | the edit, compile and test loop, with a debugger |
| `dev-fast` | Debug, `-O0` without `-g` | the fastest compile when you do not need a debugger |
| `relwithdebinfo` | RelWithDebInfo (`-O2 -g`) | an optimized build you can still debug |
| `release` | Release (`-O3`) | the whole test suite, benchmarks and production runs |

Tests are off by default: add `-DBUILD_TESTS=ON` when you configure.

### Build and run the tests

All the sequential unit tests are GoogleTest tests compiled into one executable, `test_samurai_lib`.
Run the whole suite from a `release` build:

```bash
cmake --preset release -DBUILD_TESTS=ON
cmake --build build/release --target test_samurai_lib -j2
./build/release/tests/test_samurai_lib
```

The run ends with `[  PASSED  ] 461 tests.` after less than a minute.
`-j2` keeps the memory use low; raise it if your machine has the memory for more compiler processes.

While you work on a change, use the `dev` preset and run only the tests you touch with `--gtest_filter`, for example the interval tests:

```bash
cmake --preset dev -DBUILD_TESTS=ON
cmake --build build/dev --target test_samurai_lib -j2
./build/dev/tests/test_samurai_lib --gtest_filter='interval*'
```

The whole suite takes about 40 times longer at `-O0` than in a `release` build, mostly in the `ProjectionPredictionRoundtrip` tests.

`ctest` runs only the MPI tests described below: the sequential suite is not registered with CTest, so run `test_samurai_lib` directly.
Demos are left out of the default build target unless you configure with `-DBUILD_DEMOS=ON`, so build a demo by naming its target, for example `finite-volume-advection-2d`.

### Run the demo regression tests

The `tests/test_demo_*.py` files run the demos and compare their HDF5 output with the files in `tests/reference/`.
These tests look for the demos in `build/demos/`, so configure a build in `build/` itself, build the demos you need, and run pytest from `tests/` with `--h5diff`:

```bash
cmake --preset release -B build -DBUILD_TESTS=ON
cmake --build build --target finite-volume-advection-2d -j2
cd tests
pytest -v --h5diff test_demo_finite_volume.py -k advection_2d
```

A demo that is not built makes its test fail with `FileNotFoundError`.
The CI builds every demo with `-DBUILD_DEMOS=ON` and runs `pytest -v -s --h5diff` on the whole directory.

### Test with PETSc

The implicit schemes rely on PETSc.
Install it with `pkg-config` in `samurai-env`, then configure with `-DWITH_PETSC=ON`.
This adds the `test_fv_operators_petsc` executable, which holds the PETSc tests:

```bash
conda install -c conda-forge petsc pkg-config
cmake --preset dev -DBUILD_TESTS=ON -DWITH_PETSC=ON
cmake --build build/dev --target test_fv_operators_petsc -j2
./build/dev/tests/test_fv_operators_petsc
```

The run ends with `[  PASSED  ] 42 tests.`

### Test with MPI

The MPI build needs MPI, parallel HDF5 and Boost.MPI.
`conda/mpi-environment.yml` provides them in a separate environment, `samurai-mpi-env`, together with ParMETIS.
Install the compiler there as well:

```bash
conda env create -f conda/mpi-environment.yml
conda activate samurai-mpi-env
conda install -c conda-forge cxx-compiler
```

Configure with `-DWITH_MPI=ON`, and add `-DSAMURAI_WITH_PARMETIS=ON` to test the ParMETIS partitioner.
Keep this build apart from the sequential one with `-B`:

```bash
cmake --preset dev -B build/dev-mpi -DBUILD_TESTS=ON -DWITH_MPI=ON -DSAMURAI_WITH_PARMETIS=ON
```

The MPI tests live in `tests/mpi/`.
Each file is a GoogleTest executable that CTest runs with `mpiexec` on 2, 3 and 4 processes, under the names `<test>_np2`, `<test>_np3` and `<test>_np4`.
Build one and run it with `ctest`:

```bash
cmake --build build/dev-mpi --target test_lb_graph -j2
ctest --test-dir build/dev-mpi -R test_lb_graph --output-on-failure
```

`ctest` reports `100% tests passed out of 3`.
To run an MPI test by hand, call `mpiexec` yourself:

```bash
mpiexec -n 4 ./build/dev-mpi/tests/mpi/test_lb_graph
```

## Build the documentation

The documentation is built with Sphinx, MyST and Breathe, from the sources in `docs/source/`.
The Doxygen XML of the C++ headers feeds the API pages.

1. Create and activate the documentation environment:

   ```bash
   conda env create -f docs/environment.yml
   conda activate samurai-doc
   ```

2. Build the HTML pages:

   ```bash
   cd docs
   make html
   ```

   `make html` runs Doxygen, then Sphinx.
   Open `docs/build/html/index.html` in a browser to read the result.

To run Sphinx alone, call `sphinx-build -b html source build/html` from `docs/`.
It reuses the Doxygen XML in `docs/xml/`, so run `make html` again after you change a header.
When `docs/xml/` does not exist, `conf.py` runs Doxygen first; without Doxygen, the build goes on with a warning and empty API pages.

Every pull request also gets a Read the Docs preview, linked from the checks of the pull request.

### Compile the documentation snippets

The C++ code shown in the documentation comes from compiled files: `docs/source/howto/snippet/`, `docs/source/tutorial/snippet/` and `docs/source/reference/snippet/`.
Each `.cpp` file in a subdirectory of these directories, such as `docs/source/howto/snippet/box/2d_box.cpp`, becomes an executable named after the file, so file names must be unique across the three trees.
Build them all in a separate directory, from `samurai-env`:

```bash
cmake --preset dev -B build/snippets -DBUILD_SNIPPETS=ON
cmake --build build/snippets -j2
```

### Show the output of a snippet

A page never shows a program output typed by hand.
The output of a snippet comes from a generated file next to its source, `<name>_output.txt`, which the page includes:

````markdown
```{literalinclude} snippet/mesh/uniform_output.txt
  :language: text
```
````

To show the output of a new snippet:

1. Declare it in the `CMakeLists.txt` of the snippet tree, after the snippet targets:

   ```cmake
   samurai_snippet_output(uniform)
   samurai_snippet_output(predefined_options OUTPUT predefined_options_help_output.txt ARGS --help HEAD 13)
   ```

   Give the command-line arguments with `ARGS`, and a file name with `OUTPUT` when the same program has several outputs.
   `NP`, `EXIT_CODE`, `STDERR`, `INPUTS`, `REPLACE`, `HEAD` and `TAIL` cover MPI runs, programs that fail on purpose, error messages, input files, values that change from run to run, and long outputs.
   `cmake/snippetOutputs.cmake` documents them.
   Declare PETSc snippets inside the `if(WITH_PETSC)` block.
2. Regenerate the files:

   ```bash
   cmake --build build/snippets --target update_snippet_outputs
   ```

   The target runs every declared snippet in a temporary directory and rewrites the `_output.txt` files that changed.
   Outputs declared with `NP` are generated only in a build configured with `-DWITH_MPI=ON`, the others only in a build without MPI: run the target in both builds, with `-DWITH_PETSC=ON`, to regenerate them all.
3. Include the file in the page with `literalinclude`; use `:start-at:`, `:end-before:` or `:lines:` to show part of it.
4. Commit the `_output.txt` files with the snippet.

Never edit an `_output.txt` file by hand: run `update_snippet_outputs` again when a library change alters an output.
The `compile_snippets` CI job regenerates the files and fails if one of them differs from the committed one.
The output of a snippet must be the same on every run and every machine. Leave out values that vary, such as timings or residuals near machine precision, or mask them with `REPLACE`.

### Documentation conventions

Every page under `docs/source/` is MyST Markdown.
Keep to these conventions so that the pages build and read the same way:

- Insert the project name with the `{{ project }}` substitution, not with the reStructuredText `|project|`.
- Show C++ code with a `literalinclude` directive pointing to a compiled snippet or demo, not with a code block typed in the page, so that the code shown always compiles.
- Write intervals as half-open `[a, b)`, the way samurai prints them.
- Write directives as fenced blocks (```` ```{remark} ````) and cross-references with roles such as `{doc}`, `{ref}` and `{cpp:class}`.
- Set the definition of a term with ```` ```{definition} term ```` and a side remark with ```` ```{remark} ````, not with `{note}`.
  The definitions and remarks of a page are numbered in one sequence ("Definition 1.", "Remark 2.").
  Give a definition a `:label:` to link to it with `{ref}`: the link reads "Definition 1 (term)".
- Draw a figure with code, not with a drawing program: write it as a function in `docs/source/_ext/samurai_figures/pages/<page>.py`, register it with the `@figure` decorator, and show it with a `plate` or a `diagram` directive.
  A schematic that depends on no program data may be a hand-written SVG file instead, see [Plates and diagrams](#plates-and-diagrams).

The Markdown files follow the rules in `.markdownlint.json`.
pre-commit does not run markdownlint, so run it yourself; it needs Node.js:

```bash
npx --yes markdownlint-cli2 "docs/**/*.md"
```

### Plates and diagrams

A diagram is an inline drawing with a caption.
A plate is a larger illustration in a frame, numbered in the page ("Plate 1"), with a legend that explains each of its figures:

````markdown
```{diagram}
:figure: boolean_operations

The four results on the cells 0 to 13 of level 0.
```

```{plate} One time step of the solver
:figure: time_step
:label: plate-time-step

**Fig. 1.** *adapt*: the adaptation rebuilds the mesh.
*a*, the face between a coarse cell and two fine cells.

*The plate is drawn with levels 3 to 6; the program uses levels 4 to 8.*
```
````

`:figure:` is the name of the figure function.
In a legend, start the paragraph of each figure with `**Fig. 1.**`, write the callout letters of the drawing in italics, and write a note as a paragraph all in italics.
`:columns: 1` sets the legend in one column instead of two.
`{ref}` on the label of a plate reads "Plate 1 (title)"; a diagram is not numbered, so link to its label with an explicit text, ``{ref}`the diagram <label>` ``.

A figure function takes the prefix of its SVG ids and returns a `Drawing`.
Several figures share a page, so every id it creates (patterns, markers, clip paths) starts with that prefix.
A plate also gives its `panels`: the region of each figure, which narrow screens show one under the other.
Build on `draw.py` for the primitives and on `mesh.py` for meshes.

A schematic that depends on no program data, such as a velocity set, a chain of steps or a ghost layer, may be a hand-written SVG file instead.
Replace `:figure:` with `:svg:` and the path of the file, relative to the page:

````markdown
```{plate} The D2Q9 velocity set
:svg: figures/d2q9.svg
:panels: 0 0 340 268 half; 371 0 340 268 half
```
````

The directive inlines the file in the page and prefixes its ids, so several figures can share a page.
The panels of a plate are `x y w h` regions of the drawing, separated by semicolons, each optionally followed by `half` (two side by side) or `wide` (scrolls sideways).
Give them with `:panels:` or with a `data-panels` attribute on the root `<svg>`, not both; without panels, the plate scrolls sideways on narrow screens.
The build checks the file and fails with the line and the value at fault when it breaks one of these rules:

- the root `<svg>` has `viewBox="0 0 711 H"` for a plate or `viewBox="0 0 680 H"` for a diagram, so that one unit is one CSS pixel and strokes keep their weights, and an `aria-label` that describes the figure;
- every `fill`, `stroke`, `stop-color` and `color`, in an attribute, a `style` attribute or a `<style>` rule, is `none`, `currentColor`, a `var(--sm-*)` variable of `samurai.css` or a `url(#id)` of the file;
- the file holds drawing elements only: no `<script>`, `<image>`, `<foreignObject>`, link, filter or animation;
- every `href` and `url()` points to an id of the file.

A shape without a `fill` takes `var(--sm-ink)`, and the rules of a `<style>` apply inside the figure only.
Use the `sm-fig-*` classes for text and the stroke weights of `draw.py` (0.5, 0.75, 1 and 1.5), as the generated figures do.

A figure must stay true to samurai:

- Draw what the code of the page does: the cells, intervals and levels of a figure are the ones its program prints or uses.
- Draw meshes with `build_mesh`: it makes them graded, like samurai meshes, and `check_graded` fails the build otherwise.
- Draw intervals half-open, with `bracket`: a filled dot at the start, an open dot at the end.
- When a figure uses fewer levels than the program, so that the cells stay visible, say so in the legend.
- Take every colour from the theme variables of `_static/css/samurai.css` (`var(--sm-ink)`, `var(--sm-red)`, ...), never a fixed colour, so that the figure follows the light and the dark themes.
  Add a variable there, in the light and in both dark blocks, when a figure needs a new colour.

## Check your changes with pre-commit

`.pre-commit-config.yaml` defines the hooks:

- file hygiene from `pre-commit-hooks`: final newline, trailing whitespace, merge conflict markers, case conflicts, mixed line endings (fixed to LF), YAML and JSON syntax;
- `no-commit-to-branch`, which rejects a commit on `main`: work on a branch;
- `forbid-tabs` and `remove-tabs`, which replace tabs with four spaces;
- `clang-format` on `.hpp` and `.cpp` files, with the style in `.clang-format`.

Run them all before you push:

```bash
pre-commit run --all-files
```

The CI runs the same command and fails if a hook changes a file.
When a hook fixes files, review the changes, stage them and commit again.

## Open a pull request

### Title and commits

Pull requests are squash-merged: the pull request title becomes the commit message on `main`.
The title must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `type(scope): summary`, in the imperative, lower case, with no final period.
The types are `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style` and `test`.
The `Validate PR title` check rejects a title that does not follow this form.
For example:

```text
fix(mesh): start an AMR mesh at its max level by default
docs(howto): add a how-to for restarting a simulation from a checkpoint
```

`CHANGELOG.md` is generated by [release-please](https://github.com/googleapis/release-please) from these titles at each release.
Do not edit it by hand.

### Description

The pull request template asks for three sections:

- **Description**: what was wrong or missing and what the pull request changes, with exact names of functions, files, demos and options;
- **Related issue**: `Closes #123`, or where the problem was found;
- **How has this been tested?**: the new or changed tests, the demos and commands you ran with their options, and what you did not test.

Give results as numbers or before/after comparisons, and leave out the list of changed files: the diff already shows it.

### Continuous integration

The CI first classifies the changed files.

- **Every pull request** runs `pre-commit` and the title check.
- **Documentation-only changes** (`docs/`, Markdown files, issue and pull request templates, release metadata) skip the build and test jobs.
  When they touch a snippet directory, `compile_snippets` compiles the snippets, with and without MPI, and checks their outputs.
- **Any other change** runs the full CI:
  - `cppcheck`;
  - `linux-mamba`: GCC 11 to 13 and Clang 16 to 19, with PETSc, the demos and the tests; it runs `test_samurai_lib`, `test_fv_operators_petsc` and the pytest suite;
  - `linux-mpi-mamba`: MPI demos on 1 to 9 processes and the MPI CTest tests;
  - `macos-mamba`;
  - `linux-mamba-check-nan` and `linux-mamba-check-nan-mpi`: demos built with `SAMURAI_CHECK_NAN`;
  - `compile_snippets`, which also checks the snippet outputs, since they depend on the library, and `benchmarks`.

The `ci-status` job sums up the result: it passes when every job has passed or was skipped.

### Review

A maintainer reviews the pull request and merges it once the review is settled and `ci-status` passes.
Answer the comments with new commits on the same branch: the squash merge folds them into one commit, so you do not need to rewrite history.
