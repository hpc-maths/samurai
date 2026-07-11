# yt loader for samurai files

Samurai writes its state as a **compressed HDF5 file** that stores only its
internal interval representation of the mesh (via `samurai::dump`; this is
becoming samurai's standard format, read back with `samurai::load`). The file is
compact but not directly usable, because no explicit geometry is stored.

This loader opens such a file directly in [yt](https://yt-project.org): it
rebuilds the adaptive mesh on the fly and hands it to yt as an AMR dataset, so
the full yt toolbox (slices, projections, profiles, volume rendering, derived
fields) works on samurai data.

## Files

| File                           | Role                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `samurai_load.py`              | Pure `numpy` + `h5py` reconstruction of a samurai file. No yt dependency; unit-testable. Exposes `read_cells()`, `read_amr_grids()`, `discover_series()`, `read_time()`. |
| `samurai_yt.py`                | Thin yt wrapper. Exposes `load_samurai()` and `load_samurai_series()`.                                                                       |
| `tests/generate_reference.cpp` | Regenerates the small reference files used by the tests.                                                                                     |

## Requirements

- **yt 4.x** and **`h5py`** (with `numpy`) in the same Python environment.

```bash
python -m pip install yt h5py
```

## Usage

```python
import yt
import samurai_yt

ds = samurai_yt.load_samurai("solution.h5")   # a samurai::dump file

# Anything yt can do on an AMR dataset now works:
yt.SlicePlot(ds, "z", ("stream", "u")).save()
print(ds.all_data().quantities.extrema(("stream", "u")))
```

Fields are exposed under the `"stream"` field type: a scalar field `u` becomes
`("stream", "u")`; a vector field `v` with `n` components becomes
`("stream", "v_0")`, ..., `("stream", "v_<n-1>")`.

Fields default to a **linear** color scale, which is the sensible default for the
PDE solutions samurai produces: otherwise yt guesses a log scale from the value
range and turns tiny numerical over/undershoots (and any negative value) into
misleading visual noise. Pass `take_log=True` to `load_samurai` to restore yt's
log scale, or set it per plot with `plot.set_log(("stream", "u"), True)`.

If `samurai_yt` is not on the Python path, add the plugin directory first:

```python
import sys; sys.path.insert(0, "/path/to/samurai/tools/yt")
```

### Showing the mesh

Overlay the samurai mesh on a slice with `annotate_cell_edges`:

```python
p = yt.SlicePlot(ds, "z", ("stream", "u"))
p.annotate_cell_edges()
p.zoom(8)                        # zoom in: cells get tiny in refined regions
p.save()
```

You can also color directly by refinement level with the built-in field
`("index", "grid_level")`:

```python
yt.SlicePlot(ds, "z", ("index", "grid_level")).save()
```

**Prefer `annotate_cell_edges` over `annotate_grids` here.** `annotate_grids`
draws the boundary of each yt *grid object*, not of each cell -- and the
unrefined background is a single grid object covering the whole domain (see
"How it works" below), so `annotate_grids` shows no lines at all there, which
can look like cells are missing. It isn't a bug: `annotate_cell_edges` on the
same dataset shows every cell, including the coarse background.

### Time series / animations

Write one file per output step with a numbered suffix, e.g. `sol_ite_0.h5`,
`sol_ite_1.h5`, ... (this is what the `--nfiles` option of the demos produces via
the `_ite_{n}` suffix). Then load the whole series from **any one file**:

```python
series = samurai_yt.load_samurai_series("sol_ite_0.h5")
for ds in series:
    yt.SlicePlot(ds, "z", ("stream", "u")).save(f"frame_{ds.current_time}")
```

The siblings sharing the same prefix are discovered on disk (natural order, `_2`
before `_10`). Each dataset's `current_time` is the **physical** simulation time
when the file stores it — i.e. the `"time"` HDF5 root attribute written by
samurai's `MetadataWriter` (`include/samurai/io/metadata.hpp`, `.time(t)`) — and
falls back to the iteration number otherwise. To record the physical time, pass a
metadata callback to `samurai::dump`:

```cpp
samurai::dump(path, fmt::format("sol_ite_{}", n),
              [&](samurai::MetadataWriter& m) { m.time(t); },
              mesh, u);
```

### MPI files

A file written by an MPI run stores one slice per rank. The loader reads every
rank and merges them into a single AMR hierarchy, so an MPI run and a sequential
run produce the same dataset.

### Raw cell data (without yt)

`samurai_load.read_cells()` returns the leaf cells as plain numpy arrays
(`centers`, `levels`, `fields`) with no yt dependency — handy for quick analysis
or testing.

## How it works

Samurai is **cell-based** (octree) AMR: any cell may be a leaf, and a refined
cell splits into `2**dim` children. yt's `load_amr_grids` expects **block** AMR
(Enzo style), where every grid begins on a cell edge of its *parent* level. A
lone fine cell on an odd index (the second child of its parent) does not, so
cells cannot be handed to yt one by one.

The loader therefore rebuilds the octree as block-aligned grids:

1. it reproduces the `for_each_cell` traversal (the offset-driven interval
   iterator of `LevelCellArray`) to recover each leaf's integer index and value —
   essential because field values are stored as a flat buffer in exactly that
   order, with no explicit indexing;
2. it emits one root grid covering the whole domain at `min_level` (yt level 0),
   then one `2**dim`-cell grid per refined cell, whose left edge is the parent
   cell edge — exactly what yt's alignment check wants. Grid edges are computed
   with yt's own `linspace` arithmetic so they match that check bit for bit.

Cells covered by a finer grid are masked by yt, so a coarse cell's placeholder
value is never selected — only the leaves survive, each with its own value.

## Tests

`tests/test_yt_reader.py` (run from the repository test suite) reconstructs the
reference files in `tests/reference/yt/` and checks that every cell carries the
analytic value `u = sum_d center[d]*10**d` of its own center — validating
geometry **and** traversal order simultaneously — both at the cell level (no yt)
and end-to-end through a loaded yt dataset, plus the MPI and uniform-mesh cases.

### Regenerating the reference files

Build the `generate_yt_reference` CMake target (requires `-DBUILD_TESTS=ON`) and
run it — or compile `tests/generate_reference.cpp` by hand against the samurai
headers. Example (adapt the toolchain to your environment):

```bash
mpic++ -DSAMURAI_WITH_MPI -DSAMURAI_ENABLE_INLINE \
  -DSAMURAI_FIELD_CONTAINER_XTENSOR -DSAMURAI_FLUX_CONTAINER_XTENSOR \
  -DSAMURAI_STATIC_MAT_CONTAINER_XTENSOR -DFMT_SHARED \
  -I../../../include -std=c++20 \
  tests/generate_reference.cpp -o generate_reference \
  -lhdf5 -lpugixml -lfmt -lboost_mpi -lboost_serialization

./generate_reference ../../../tests/reference/yt             # ref_1d/2d/3d/uniform.h5
mpirun -n 2 ./generate_reference ../../../tests/reference/yt --mpi  # ref_2d_mpi.h5
```

`.h5` files are git-ignored globally, so the reference fixtures must be added
with `git add -f`.

## Current limitations

- A samurai file only contains the leaf cells (`mesh_id_t::cells`). Other
  submeshes (ghosts, `by_mesh_id`) are not reconstructible from it.
- The root grid is dense at `min_level`: a very large `min_level` makes it heavy.
  Typical samurai meshes use a small `min_level`, so this is rarely an issue.
- The domain is assumed rectangular (the common case). A hole in the coarse
  covering would read back as an empty region rather than an error.
