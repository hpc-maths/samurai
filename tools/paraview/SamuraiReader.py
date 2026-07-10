# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""
ParaView reader for samurai data files (compressed HDF5).

Load it through ParaView's *Tools > Manage Plugins... > Load New...* and pick
this file.  A samurai data file only stores the compressed interval representation of
the mesh; the finite-element view (quadrilaterals in 2D, hexahedra in 3D) is
rebuilt on the fly by :mod:`samurai_load`.  With an MPI file, one partition is
produced per rank.

Time series / animation: open **any single file** of a numbered series (e.g.
``sol_ite_0.h5``, ``sol_ite_1.h5``, ...).  The reader auto-discovers the sibling
files on disk, so it does not rely on ParaView's file-series grouping: every file
becomes one time step and the time controls animate through them.

Requires ``h5py`` in ParaView's Python (see README.md).
"""

import os
import sys

import numpy as np

# Make ``samurai_load`` importable regardless of ParaView's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from samurai_load import load, discover_series  # noqa: E402

from paraview.util.vtkAlgorithm import (  # noqa: E402
    VTKPythonAlgorithmBase,
    smproxy,
    smproperty,
    smdomain,
    smhint,
)
from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet, vtkUnstructuredGrid  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkDataObject  # noqa: E402
from vtkmodules.vtkCommonCore import vtkPoints  # noqa: E402
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline  # noqa: E402
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray  # noqa: E402
from vtkmodules.util.vtkConstants import VTK_UNSIGNED_CHAR  # noqa: E402


def _build_unstructured_grid(block):
    """Turn a :func:`samurai_load.load` block into a ``vtkUnstructuredGrid``."""
    points = np.ascontiguousarray(block["points"], dtype=np.float64)
    connectivity = block["connectivity"]
    n_cells, n_vertices = connectivity.shape

    ug = vtkUnstructuredGrid()

    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=1))
    ug.SetPoints(vtk_points)

    if n_cells > 0:
        # vtkCellArray (VTK 9) = offsets + flat connectivity.
        offsets = np.arange(0, (n_cells + 1) * n_vertices, n_vertices, dtype=np.int64)
        conn = np.ascontiguousarray(connectivity.ravel(), dtype=np.int64)

        from vtkmodules.vtkCommonDataModel import vtkCellArray

        cell_array = vtkCellArray()
        cell_array.SetData(
            numpy_to_vtkIdTypeArray(offsets, deep=1),
            numpy_to_vtkIdTypeArray(conn, deep=1),
        )
        cell_types = np.full(n_cells, block["vtk_cell_type"], dtype=np.uint8)
        ug.SetCells(numpy_to_vtk(cell_types, deep=1, array_type=VTK_UNSIGNED_CHAR), cell_array)

    cell_data = ug.GetCellData()

    def add_array(name, values, active_scalar=False):
        arr = numpy_to_vtk(np.ascontiguousarray(values), deep=1)
        arr.SetName(name)
        cell_data.AddArray(arr)
        if active_scalar and values.ndim == 1:
            cell_data.SetActiveScalars(name)

    # The first field is set as the active scalar so ParaView colors by it by
    # default; the user can always switch the coloring array afterwards.
    first = True
    for name, values in block["fields"].items():
        v = values[:, 0] if values.shape[1] == 1 else values
        add_array(name, v, active_scalar=first)
        first = False

    for extra in ("level", "indices", "center"):
        if extra in block:
            add_array(extra, block[extra])

    return ug


@smproxy.reader(
    name="SamuraiReader",
    label="Samurai reader",
    extensions="h5",
    file_description="Samurai data (HDF5)",
)
class SamuraiReader(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkPartitionedDataSet")
        self._filename = None
        self._extra_arrays = True
        self._series_cache = None  # (filename, files, times)

    # -- File name: open any one file; the numbered series is auto-discovered. --
    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="h5", file_description="Samurai data (HDF5)")
    def SetFileName(self, name):
        if name != self._filename:
            self._filename = name
            self._series_cache = None  # invalidate the discovered series
            self.Modified()

    def _series(self):
        """Discovered ``(files, times)`` for the current file, memoized.

        ``GetTimestepValues`` is polled often by the GUI and ``discover_series``
        lists the directory and opens every sibling file, so the result is cached
        until the file name changes.
        """
        if not self._filename:
            return [], []
        if self._series_cache is None or self._series_cache[0] != self._filename:
            files, times = discover_series(self._filename)
            self._series_cache = (self._filename, files, times)
        return self._series_cache[1], self._series_cache[2]

    # Exposes the time controls in the GUI and populates the animation.
    @smproperty.doublevector(
        name="TimestepValues",
        information_only="1",
        si_class="vtkSITimeStepsProperty",
    )
    def GetTimestepValues(self):
        _, times = self._series()
        return times or None

    @smproperty.intvector(name="ExposeMeshArrays", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>')
    def SetExposeMeshArrays(self, value):
        value = bool(value)
        if value != self._extra_arrays:
            self._extra_arrays = value
            self.Modified()

    # -- Pipeline ---------------------------------------------------------------
    def RequestInformation(self, request, inInfoVec, outInfoVec):
        _, times = self._series()
        if not times:
            return 1

        info = outInfoVec.GetInformationObject(0)
        info.Remove(vtkStreamingDemandDrivenPipeline.TIME_STEPS())
        info.Remove(vtkStreamingDemandDrivenPipeline.TIME_RANGE())
        for t in times:
            info.Append(vtkStreamingDemandDrivenPipeline.TIME_STEPS(), t)
        info.Append(vtkStreamingDemandDrivenPipeline.TIME_RANGE(), times[0])
        info.Append(vtkStreamingDemandDrivenPipeline.TIME_RANGE(), times[-1])
        return 1

    def _requested_index(self, info, times):
        if not times:
            return 0
        if info.Has(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()):
            t = info.Get(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
            # nearest advertised time
            return min(range(len(times)), key=lambda i: abs(times[i] - t))
        return 0

    def RequestData(self, request, inInfoVec, outInfoVec):
        if not self._filename:
            raise RuntimeError("No file name set for SamuraiReader")

        files, times = self._series()
        info = outInfoVec.GetInformationObject(0)
        index = self._requested_index(info, times)
        filename = files[index]

        blocks = load(filename, extra_arrays=self._extra_arrays)
        output = vtkPartitionedDataSet.GetData(outInfoVec, 0)
        output.SetNumberOfPartitions(len(blocks))
        for p, block in enumerate(blocks):
            output.SetPartition(p, _build_unstructured_grid(block))

        output.GetInformation().Set(vtkDataObject.DATA_TIME_STEP(), float(times[index]))
        return 1
