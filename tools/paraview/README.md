# ParaView reader for samurai files

Samurai writes its state as a **compressed HDF5 file** that stores only its
internal interval representation of the mesh (via `samurai::dump`; this is
becoming samurai's standard format, read back with `samurai::load`). The file is
compact but not directly viewable, because no explicit geometry is stored.

This plugin lets ParaView open such a file directly: it rebuilds the
finite-element view (points, quadrilaterals in 2D / hexahedra in 3D, cell data)
on the fly. It replaces the old explicit-mesh export (points + connectivity +
`.xdmf`), which is being retired.

## Files

| File | Role |
|------|------|
| `samurai_load.py` | Pure `numpy` + `h5py` reconstruction of a samurai file. No ParaView dependency; unit-testable. Exposes `load()`, `discover_series()`, `read_time()`. |
| `SamuraiReader.py` | ParaView reader (`@smproxy.reader`) wrapping `samurai_load`. |
| `tests/generate_reference.cpp` | Regenerates the small reference files used by the tests. |

## Requirements

The reader needs **`h5py`** (and `numpy`, always bundled) in ParaView's Python.

Check inside ParaView's Python shell (*View > Python Shell*):

```python
import h5py  # must succeed
```

If it fails, install it into ParaView's interpreter. Recent ParaView binaries
ship a `pip`:

```bash
# adjust to your ParaView install
/path/to/ParaView.app/Contents/bin/pvpython -m pip install h5py
```

## Usage

1. *Tools > Manage Plugins... > Load New...* and select `SamuraiReader.py`
   (optionally tick *Auto Load*).
2. *File > Open* and choose a samurai `.h5` file (e.g. `mysolution.h5` produced
   by `samurai::dump`). The reader is registered for the `.h5` extension.
3. Apply. Color by any field.

Options in the *Properties* panel:

- **Expose mesh arrays** (on by default): also attach `level`, `indices` and
  `center` as cell arrays. They are computed for free during reconstruction and
  are handy to filter/color by AMR level (equivalent to `--save-debug-fields`).

### Time series / animations

To animate a simulation, write one file per output step with a numbered suffix,
e.g. `sol_ite_0.h5`, `sol_ite_1.h5`, ... (this is what the `--nfiles` option of
the demos produces via the `_ite_{n}` suffix).

Then simply **open any one file of the series** (e.g. `sol_ite_0.h5`). The reader
auto-discovers the sibling files on disk (same prefix, trailing number) and
exposes one time step per file — it does **not** rely on ParaView's file-series
grouping. The usual VCR / time controls then step or play through them, and
*File > Save Animation* exports a movie.

Files are ordered naturally (`_2` before `_10`). The **time value** shown by
ParaView is the physical simulation time when the file stores it — i.e. the
`"time"` HDF5 root attribute written by samurai's `MetadataWriter`
(`include/samurai/io/metadata.hpp`, `.time(t)`). If any file of the series lacks
it, the reader falls back to the trailing iteration number. A file with no
trailing number is a single snapshot (one time step).

To record the physical time, pass a metadata callback to `samurai::dump`:

```cpp
samurai::dump(path, fmt::format("sol_ite_{}", n),
              [&](samurai::MetadataWriter& m) { m.time(t); },
              mesh, u);
```

### MPI files

A file written by an MPI run stores one slice per rank. The reader reconstructs
each rank independently and returns a `vtkPartitionedDataSet` with one partition
per rank.

## How it works

A samurai file stores, per level and per direction, the compressed intervals
(`m_cells`) and offsets (`m_offsets`) plus `dim`, `min/max_level`,
`origin_point` and `scaling_factor`. No geometry is stored. The reader:

1. rebuilds the integer cell indices `(i, j, k)` by reproducing the
   `for_each_cell` traversal (the offset-driven interval iterator of
   `LevelCellArray`) — this is essential because field values are stored as a
   flat buffer in exactly that order, with no explicit indexing;
2. computes the geometry with `length = scaling_factor / 2**level`,
   `corner = origin_point + length * indices`, and the unit-cell vertex offsets
   of samurai's `get_element`;
3. attaches each field as **cell data**.

## Tests

`tests/test_paraview_reader.py` (run from the repository test suite) reconstructs
the reference files in `tests/reference/paraview/` and checks that every cell
carries the analytic value `u = sum_d center[d]*10**d` of its own center —
validating geometry **and** traversal order simultaneously — plus the MPI
partition and time-series invariants.

### Regenerating the reference files

They are produced by `tests/generate_reference.cpp`, compiled against the
samurai headers. Example (adapt the toolchain to your environment):

```bash
mpic++ -DSAMURAI_WITH_MPI -DSAMURAI_ENABLE_INLINE \
  -DSAMURAI_FIELD_CONTAINER_XTENSOR -DSAMURAI_FLUX_CONTAINER_XTENSOR \
  -DSAMURAI_STATIC_MAT_CONTAINER_XTENSOR -DFMT_SHARED \
  -I../../../include -std=c++20 \
  tests/generate_reference.cpp -o generate_reference \
  -lhdf5 -lpugixml -lfmt -lboost_mpi -lboost_serialization

./generate_reference ../../../tests/reference/paraview            # ref_1d/2d/3d.h5
mpirun -n 2 ./generate_reference ../../../tests/reference/paraview --mpi  # ref_2d_mpi.h5
```

## Current limitations

- A samurai file only contains the leaf cells (`mesh_id_t::cells`). Other
  submeshes (ghosts, `by_mesh_id`) are not reconstructible from it. The `level`
  cell array lets you filter by AMR level instead.
- Points are emitted per cell (no deduplication). This is correct for
  visualization; deduplication is a possible future memory optimization.
