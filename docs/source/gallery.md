# samurai gallery

The [samurai gallery](https://hpc-maths.github.io/samurai-gallery/) shows complete simulations run with {{ project }}.
Each case has a page with the equations, the numerical method, the source code and a video of the solution with its adapted mesh.
Pick the case closest to your problem, build it, and change it from there.

The gallery lives in its own repository, [hpc-maths/samurai-gallery](https://github.com/hpc-maths/samurai-gallery).
For shorter programs that come with the {{ project }} sources, see the {doc}`demos <demos>`.

## Cases

The cases are grouped by category, as in the `cases/<category>/<name>/` folders of the repository.
All of them are 2D or 1D finite volume simulations on a mesh adapted by multiresolution.

### Getting started

| Case | What it shows |
| --- | --- |
| [Linear advection of a disc](https://hpc-maths.github.io/samurai-gallery/cases/getting-started/advection-2d/) | A disc is transported diagonally while the multiresolution mesh tracks its edges. The {doc}`getting started tutorial <tutorial/getting_started>` explains the same problem line by line. |

### Hyperbolic

| Case | What it shows |
| --- | --- |
| [Burgers equation (1D)](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/burgers-1d/) | A smooth hat steepens into a shock, and the mesh follows the front. |
| [Burgers equation (2D)](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/burgers-2d/) | A radial hat steepens into nonlinear fronts. |
| [Double Mach reflection](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/double-mach-reflection/) | A Mach 10 shock strikes a wedge and produces the double Mach reflection pattern. |
| [Isentropic vortex](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/isentropic-vortex/) | A smooth vortex is advected across the domain; the exact solution is known. |
| [Kelvin-Helmholtz instability](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/kelvin-helmholtz/) | Two shear layers roll up into vortices that the mesh tracks as they grow. |
| [2D Riemann problem (config 3)](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/riemann-2d/) | Four constant states meet at a point and interact through shocks, contacts and a jet. |
| [2D Riemann problem (config 4)](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/riemann-2d-config4/) | Four constant states interact through shocks and contacts. |
| [2D Riemann problem (config 12)](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/riemann-2d-config12/) | Four constant states interact through shocks and contacts. |
| [Sedov blast wave](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/sedov-blast/) | A point release of energy drives a circular blast wave, and the mesh refines on the shock. |
| [Sod shock tube](https://hpc-maths.github.io/samurai-gallery/cases/hyperbolic/sod-shock-tube/) | The Sod shock tube, rotated into 2D: a rarefaction, a contact and a shock. |

### Interfaces

| Case | What it shows |
| --- | --- |
| [Level-set in a vortex flow](https://hpc-maths.github.io/samurai-gallery/cases/interfaces/level-set-2d/) | A swirling flow stretches a circular interface into a filament. The {doc}`level-set tutorial <tutorial/level_set>` covers the method. |
| [Two-scale isothermal atomization](https://hpc-maths.github.io/samurai-gallery/cases/interfaces/two-scale-capillarity/) | An air jet shears a liquid column that breaks up, with mass transfer from the resolved interface to a disperse phase. |

### Transport

| Case | What it shows |
| --- | --- |
| [Linear convection around an obstacle](https://hpc-maths.github.io/samurai-gallery/cases/transport/linear-convection-obstacle/) | A square profile is carried diagonally past a solid obstacle. |

## What a case contains

Each case is a self-contained folder, `cases/<category>/<name>/`.
Its `case.yaml` file holds the metadata (title, equation, dimension, adaptation), the command-line options of each run, and the version of the code the case is tested against:

- a case with its own `main.cpp` and `CMakeLists.txt` pins {{ project }} with `samurai.ref` (a tag, a commit or a branch). The gallery builds this exact version of {{ project }} and links the case against it.
- a case that runs an external solver built on {{ project }}, such as the compressible Euler cases from [samurai-euler](https://github.com/hpc-maths/samurai-euler), has an `engine` block instead: `engine.repo` and `engine.ref` give the solver and its version, and the solver's own environment provides {{ project }}. These cases also list `mpi` under `requires`.

Each case page on the site states the version it was tested with.
Read the `case.yaml` of a case before you build it against another version of {{ project }}: a case is only checked against the version it pins.

## Build and run a case

The gallery repository builds, runs and renders a case with one script.

1. Clone the repository and create its micromamba environment, which provides the compilers, the dependencies of {{ project }}, and the Python packages that render the media:

   ```bash
   git clone https://github.com/hpc-maths/samurai-gallery.git
   cd samurai-gallery
   micromamba create -f environment.yml
   micromamba activate samurai-gallery
   ```

2. Run the case with the `ci` profile, the short low-resolution run:

   ```bash
   bash infra/build_case.sh cases/getting-started/advection-2d ci
   ```

   The script builds the {{ project }} version pinned in `case.yaml` into a cache under `~/.cache/samurai-gallery`, then builds the case, runs it and renders the media.
   You do not need to install {{ project }} yourself.
   Replace `ci` with `hero` for the high-resolution run.

When the script ends, the case folder contains `thumbnail-light.png`, `thumbnail-dark.png`, `preview-light.mp4` and `preview-dark.mp4`, and the HDF5 output of the run is in its `.output/` subfolder.

A case with a `main.cpp` is also a plain CMake project that calls `find_package(samurai CONFIG REQUIRED)`.
If {{ project }} is installed with conda, as in the {ref}`installation guide <install-with-conda>`, at the version that `samurai.ref` names, you can build the case without the gallery script:

```bash
cd cases/getting-started/advection-2d
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build
./build/gallery-advection-2d --Tf 0.6 --min-level 4 --max-level 8
```

The target name and the options of each run are in the `run` block of `case.yaml`.

## Add a case

The [contributing guide of the gallery](https://github.com/hpc-maths/samurai-gallery/blob/main/CONTRIBUTING.md) explains how to scaffold a case with `scripts/new-case.py`, fill in its files, test it with `infra/build_case.sh`, and open a pull request.
The gallery CI validates the case, builds it, runs it and renders its media.
