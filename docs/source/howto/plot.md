# How-to: plot samurai fields and meshes

This guide shows how to look at the meshes and fields that a samurai program writes to disk.
samurai writes two kinds of HDF5 files, and each kind has its own viewers:

| File written by | Dimension | Viewer |
|---|---|---|
| `samurai::save` (`.h5` + `.xdmf`) | 2D, 3D | ParaView, through the `.xdmf` file |
| `samurai::save` (`.h5`) | 1D | `python/read_mesh.py` (Matplotlib) |
| `samurai::dump` (`.h5`) | 1D, 2D, 3D | ParaView with `tools/paraview/SamuraiReader.py`, or yt with `tools/yt/samurai_yt.py` |

## Before you start

- Your program saves its results with `samurai::save` or `samurai::dump`.
  If it does not, follow the [save how-to guide](save.md) first.
- You have a copy of the samurai repository.
  The scripts below are in its source tree; they are not installed with the library.
- To plot a time series, your program writes one file per output step, with the step number at the end of the name.
  The demos do this with the `--nfiles` option: `FV_advection_1d_ite_0.h5`, `FV_advection_1d_ite_1.h5`, ...

The commands use these placeholders:

- `<samurai-dir>`: the root of the samurai repository.
- `<file>`: a file name without the `.h5` extension, for example `FV_advection_1d`.
- `<prefix>`: the common part of the names in a series, up to the step number, for example `FV_advection_1d_ite_`.
- `<field>`: a field name stored in the file, for example `u`.

## Plot 2D and 3D results in ParaView

`samurai::save` writes an `.xdmf` file next to the `.h5` file.
ParaView opens it with its built-in XDMF reader.

1. In ParaView, choose *File > Open* and select `<file>.xdmf`.
2. Click *Apply*.
3. Choose the field to color by in the toolbar.

A vector field `v` with `n` components appears as `n` scalar fields `v_0`, ..., `v_<n-1>`.
To see the level of each cell, run your program with `--save-debug-fields`: the file then also holds the `levels` field and the `coordinates` and `indices` vector fields.

## Open restart files in ParaView or yt

`samurai::dump` stores only the compressed interval representation of the mesh, without points or connectivity.
Two readers in the repository rebuild the cells from it.

### ParaView reader

The ParaView reader needs ParaView 5.10 or later, with `h5py` installed in ParaView's Python.

1. Choose *Tools > Manage Plugins... > Load New...* and select `<samurai-dir>/tools/paraview/SamuraiReader.py`.
   Tick *Auto Load* to load it in every session.
2. Choose *File > Open*, select a `.h5` file written by `samurai::dump`, and pick *Samurai reader* if ParaView asks for a reader.
3. Click *Apply* and color by any field.

To animate a series, open any one file of it, for example `<prefix>0.h5`.
The reader finds the other files of the series on disk and exposes one time step per file.
A file written by an MPI run opens with one partition per rank.

The reader refuses files written by `samurai::save`; open those through their `.xdmf` file.
See the [ParaView reader README](https://github.com/hpc-maths/samurai/tree/main/tools/paraview) for the reader options and troubleshooting.

### yt loader

The yt loader needs yt 4 and `h5py` in the same Python environment.
It loads a `samurai::dump` file as a yt adaptive mesh refinement (AMR) dataset, so slices, projections and profiles work on samurai data:

```python
import sys
sys.path.insert(0, "<samurai-dir>/tools/yt")

import yt
import samurai_yt

ds = samurai_yt.load_samurai("<file>.h5")
yt.SlicePlot(ds, "z", ("stream", "<field>")).save()
```

Fields appear under the `"stream"` field type.
`samurai_yt.load_samurai_series("<prefix>0.h5")` loads a whole series as a list of datasets.
See the [yt loader README](https://github.com/hpc-maths/samurai/tree/main/tools/yt) for the mesh overlay, time values and MPI files.

## Plot 1D results with read_mesh.py

`<samurai-dir>/python/read_mesh.py` reads a 1D file written by `samurai::save` and plots it with Matplotlib.
It draws each field as a curve of its values at the cell centers.

### Install the dependencies

The script needs these Python packages:

- `h5py`
- `matplotlib`

To save a time series as a video, you also need [FFmpeg](https://ffmpeg.org) on your `PATH`: Matplotlib calls the `ffmpeg` program to write the MP4 file.

### Plot the mesh

Pass the file name without the `.h5` extension:

```bash
python <samurai-dir>/python/read_mesh.py <file>
```

A Matplotlib window opens with the cell boundaries marked along the x axis.

### Plot fields

Give the field names after `--field`.
Each field gets its own subplot, side by side:

```bash
python <samurai-dir>/python/read_mesh.py <file> --field u level
```

List all the names after a single `--field`: if you repeat the option, only the last one is kept.

If a name is not in the file, the script stops and lists the fields that are:

```text
ValueError: file:<file>> field:v not in available fields values: level u
```

### Plot a time series

To animate a series, pass `<prefix>` as the file name, and give the range of steps with `--start` and `--end`:

```bash
python <samurai-dir>/python/read_mesh.py FV_advection_1d_ite_ --field u --start 0 --end 10 --wait 100
```

The script reads `<prefix><n>.h5` for each `n` from `--start` up to `--end - 1`: here `FV_advection_1d_ite_0.h5` to `FV_advection_1d_ite_9.h5`.
`--start` defaults to 0.
The title of the figure shows the step number.

`--wait` sets the time between two frames, in milliseconds (200 by default).
The animation loops until you close the window.

### Plot results of an MPI run

`samurai::save` called in an MPI run writes one file that holds one part per rank.
`read_mesh.py <file>` plots all the parts of such a file together, with no extra option.

`--mpi-size <number-of-ranks>` only changes how the script reads a time series: it then reads one file per rank and per step, named `<prefix><n>_rank_<r>.h5`.

```{note}
`samurai::save` does not write files with these per-rank names, and a time series of files written by an MPI run fails with or without `--mpi-size`.
To animate the results of an MPI run, write them with `samurai::dump` and open them with the ParaView reader or yt.
```

### Save the plot to a file

Add `--save` with an output name, without extension, to write the plot to a file instead of opening a window:

```bash
python <samurai-dir>/python/read_mesh.py <file> --field u --save u_plot
```

The script writes `u_plot.png`, at 300 dots per inch.
For a time series, it writes an MP4 video instead:

```bash
python <samurai-dir>/python/read_mesh.py FV_advection_1d_ite_ --field u --start 0 --end 10 --save u_movie
```

This command writes `u_movie.mp4`, with one frame per step and a frame rate of 1000 / `--wait` frames per second.
Without FFmpeg, it fails with `ValueError: unknown file extension: .mp4`.

### All options

`--help` lists the options of the script:

```bash
python <samurai-dir>/python/read_mesh.py --help
```

| Option | Default | Effect |
|---|---|---|
| `<file>` or `<prefix>` | required | File name, or series prefix with `--end`, without `.h5` |
| `--field <field> [<field> ...]` | none: plot the mesh | Fields to plot, one subplot each |
| `--start <n>` | 0 | First step of a time series |
| `--end <n>` | none: plot one file | Step after the last one of a time series |
| `--wait <ms>` | 200 | Time between two frames, in milliseconds |
| `--mpi-size <number-of-ranks>` | 1 | Number of per-rank files to read for each step of a time series |
| `--save <output>` | none: open a window | Write `<output>.png`, or `<output>.mp4` for a time series |

## Related

- [Save how-to guide](save.md): write the files that this guide plots.
- [Options how-to guide](options.md): the command-line options of samurai programs.
