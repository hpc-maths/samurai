# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

The documentation is a Sphinx 8 site built from the MyST Markdown pages of `docs/source/`.
Breathe turns the Doxygen XML of the C++ headers into the API pages.
The theme is Furo, restyled by one project stylesheet, `docs/source/_static/css/samurai.css`, and one template override, `docs/source/_templates/page.html`, which draws the site header.
`conf.py` maps Furo's color and font variables onto the `--sm-*` tokens of that stylesheet, so one set of tokens drives the light and the dark themes.
Two project extensions in `docs/source/_ext/` add the numbered definitions and remarks (`statements.py`) and the plates and diagrams drawn from code (`plates.py` and `samurai_figures/`).
Read the Docs hosts one build per release with a version switcher, search within a version, and a preview for every pull request.
The homepage is the landing page of the documentation.

## Users

- Solver authors are the primary readers: applied mathematicians and CFD engineers who write finite volume or lattice Boltzmann solvers on adaptive Cartesian meshes, with multiresolution or AMR, sequential or with MPI. They come to the docs to write, build and run a solver, and return for reference material while doing it.
- Evaluators decide whether samurai fits their problem, often comparing it with other adaptive mesh libraries such as p4est, AMReX or deal.II. They need to understand the data structure and its trade-offs quickly.

## Product Purpose

samurai (Structured Adaptive mesh and MUlti-Resolution based on Algebra of Intervals) is a header-only C++20 library for numerical simulation on adaptive Cartesian meshes. It stores each level of the mesh as sets of integer intervals and expresses mesh adaptation and discretization as a set algebra over those intervals. Success means a solver author goes from install to a running adaptive simulation quickly, and an evaluator can tell from the docs whether the library fits their problem.

## Positioning

- The mesh is stored as integer intervals per level, not as a tree of cells. Schemes select whole groups of cells with set operations (intersection, union, difference, change of level) and walk them in one ordered pass, instead of searching neighbors cell by cell.
- Multiresolution and AMR run on the same interval structure.
- Adoption is cheap: the library is header-only C++20, installs from conda-forge in one command, and MPI is optional.

## Operating Context

- Readers work in a terminal and an editor with C++ code, CMake, conda and often MPI. Code samples are copied and run as is.
- The documentation is organized as tutorials, how-to guides, a philosophy page, reference pages, generated API pages, LBM pages, demos and a gallery index.
- The samurai gallery (hpc-maths.github.io/samurai-gallery, separate repository) holds complete simulation cases with videos. It is out of scope for design work here; the docs link to it.
- The README is the GitHub front page: logo, badges, an annotated figure (`docs/source/_static/readme_advection_2d.png`) and a runnable example.

## Capabilities and Constraints

- Finite volume and lattice Boltzmann schemes, multiresolution with error control, AMR, MPI with load balancing, HDF5 output.
- `CONTEXT.md` is the authoritative glossary. Docs and UI copy follow it: "level", "interval", "cell", "box", "domain" have fixed meanings, bare "projection" and "operator" are banned, and stencils are expressed as a radius.
- Half-open intervals are written `[a, b)`.
- All docs are in English, US spelling. No em dashes.
- C++ code on a page comes from a compiled snippet or demo through `literalinclude`, and program output from a generated `_output.txt` file, never typed by hand (`docs/CONTRIBUTING.md`, "Build the documentation").
- Design scope: the Sphinx docs site, the homepage inside it, and the README / GitHub presence.

## Illustrations

The rules of `docs/CONTRIBUTING.md`, "Plates and diagrams", bind every figure:

- Figures are code. A figure that shows program data (meshes, intervals, values, ranks) is a Python function in `docs/source/_ext/samurai_figures/pages/<page>.py`, registered with `@figure` and shown with a `plate` or a `diagram` directive. A schematic that depends on no program data may be a hand-written SVG passed with `:svg:`; the build checks that it uses only the `--sm-*` colours. No raster export for a new figure.
- A diagram is an inline drawing with an italic caption, not numbered. A plate is a framed, numbered illustration ("Plate 1") with a legend under "Explanation of Plate 1" that names each figure ("Fig. 1.") and each lettered callout.
- Figures are true to samurai: the cells, intervals and levels drawn are the ones the program of the page prints or uses; meshes come from `build_mesh` and are graded; intervals are half-open brackets, a filled dot at the start and an open dot at the end. When a figure uses fewer levels than the program, the legend says so.
- Every color is a theme variable of `samurai.css` (`var(--sm-ink)`, `var(--sm-red)`, ...), never a fixed value, so figures follow the light and the dark themes. A new color is added there, in the light block and in both dark blocks.

## Brand Commitments

- The name is always written lowercase: samurai.
- The logo stays as the identity mark: `docs/source/logo/` (light and dark variants, `logo.svg`), also copied in `docs/source/_static/`.
- The name is an acronym. Apart from the logo, the site uses no Japanese folklore motifs (brush strokes, ink washes, katanas, decorative patterns), and imagery tied to militarism, such as the Rising Sun flag, stays out entirely.
- Binding reference: the site reads like a book with illustrations, in the spirit of <https://ai-for-dev.github.io/combo/> (its reading structure and feel, not its illustrations). Bold primary color fields with heavy black rules (a neoplastic, De Stijl look) are off-brand.

## Evidence on Hand

- Gallery videos of solutions and adapted meshes from samurai-gallery, cleared for use on the homepage and docs.
- The annotated advection figure in the README (13,333 cells with multiresolution against 65,536 for a uniform mesh at level 8), backed by the program shown next to it.
- Absent, never to be invented: scaling or performance benchmarks, named users or adopting organizations, testimonials. Papers that use samurai are cited in the reference pages, but they are not selected as homepage proof.

## Product Principles

1. Every claim is backed by something a reader can run or check: a program, a figure with its numbers, a video.
2. Show the mechanism. The interval data structure is the reason to choose samurai, so pages explain it with figures, not slogans.
3. State the costs next to the benefits, as the philosophy page does, so evaluators can judge fit honestly.
4. Copy-paste works. Code and commands on any page run as written.
5. Words follow `CONTEXT.md`, with one term per concept across docs, homepage and README.
