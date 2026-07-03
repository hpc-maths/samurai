import h5py
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="Compare two h5 files.")
parser.add_argument(
    "file1", type=str, help="first hdf5 file to compare without .h5 extension"
)
parser.add_argument(
    "file2", type=str, help="second hdf5 file to compare without .h5 extension"
)
parser.add_argument(
    "--start", type=int, required=False, default=None, help="iteration start"
)
parser.add_argument(
    "--end", type=int, required=False, default=None, help="iteration end"
)
parser.add_argument(
    "--tol",
    type=float,
    required=False,
    default=1e-12,
    help="absolute tolerance for field comparison",
)
parser.add_argument(
    "--rtol",
    type=float,
    required=False,
    default=0.0,
    help="relative tolerance for field comparison (combined with --tol as atol: "
    "a field value differs if |v1-v2| > tol + rtol*|v2|, same convention as "
    "numpy.isclose)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="print more information about the comparison",
)
args = parser.parse_args()


def construct_cells(mesh):
    if "points" in mesh.keys():
        points = mesh["points"]
        conn = mesh["connectivity"]
        return points[:][conn[:]]
    else:
        output = None
        for k in mesh.keys():
            points = mesh[k]["points"]
            conn = mesh[k]["connectivity"]
            if output is None:
                output = points[:][conn[:]]
            else:
                output = np.concatenate((output, points[:][conn[:]]))
        return output


def construct_fields(mesh):
    if "points" in mesh.keys():
        if "fields" not in mesh.keys():
            return {}
        return mesh["fields"]
    else:
        output = {}
        for k in mesh.keys():
            if "fields" in mesh[k]:
                for f in mesh[k]["fields"].keys():
                    if f not in output.keys():
                        output[f] = mesh[k]["fields"][f][:]
                    else:
                        output[f] = np.concatenate((output[f], mesh[k]["fields"][f][:]))
        return output


def compare_meshes(file1, file2, tol, rtol=0.0):
    mesh1 = h5py.File(file1, "r")["mesh"]
    mesh2 = h5py.File(file2, "r")["mesh"]
    cells1 = construct_cells(mesh1)
    cells2 = construct_cells(mesh2)

    index1 = np.argsort(np.asarray([c.tobytes() for c in cells1]))
    index2 = np.argsort(np.asarray([c.tobytes() for c in cells2]))

    if np.any(cells1.shape != cells2.shape):
        print("shape are not compatibles")
        print(f"{cells1.shape} vs {cells2.shape}")
        sys.exit(f"files {file1} and {file2} are different")

    if np.any(cells1[index1] != cells2[index2]):
        print("cells are not the same")
        sys.exit(f"files {file1} and {file2} are different")

    field1 = construct_fields(mesh1)
    field2 = construct_fields(mesh2)

    check = True
    extra_fields = set(field2.keys()) - set(field1.keys())
    if extra_fields:
        print(f"extra fields in second file: {sorted(extra_fields)}")
        sys.exit(f"files {file1} and {file2} are different")

    for field in field1.keys():
        v1 = field1[field][:][index1]
        v2 = field2[field][:][index2]
        exact_compare = np.issubdtype(v1.dtype, np.integer) and np.issubdtype(
            v2.dtype, np.integer
        )

        if exact_compare:
            mismatch = v1 != v2
            if np.any(mismatch):
                ind = np.where(mismatch)[0]
                centers = cells1[index1[ind]].mean(axis=1)
                worst = 0
                print(
                    f"{field} is not the same between {file1} and {file2}: "
                    f"{len(ind)}/{len(v1)} cells differ  "
                    "exact integer comparison"
                )
                if args.verbose:
                    print(
                        f"{'idx':>8s}  {'center (x,y,z)':>30s}  "
                        f"{'value 1':>22s}  {'value 2':>22s}  note"
                    )
                    for i, c in zip(ind, centers):
                        print(
                            f"{i:8d}  ({c[0]: .8f},{c[1]: .8f},{c[2]: .8f})  "
                            f"{v1[i]!s:>22s}  {v2[i]!s:>22s}  exact mismatch"
                        )
                check = False
            continue

        abs_diff = np.abs(v1 - v2)
        # Ensure NaN vs non-NaN mismatches are detected (NaN comparisons are always False)
        nan_mismatch = np.isnan(v1) ^ np.isnan(v2)
        if np.any(nan_mismatch):
            abs_diff = np.where(nan_mismatch, np.inf, abs_diff)
        # relative diff normalized by the larger of the two magnitudes, so it stays
        # meaningful (and bounded) even when v1 and v2 have opposite signs
        denom = np.maximum(np.abs(v1), np.abs(v2))
        rel_diff = np.divide(
            abs_diff, denom, out=np.zeros_like(abs_diff), where=denom > 0
        )
        # scale of this field, to tell "genuinely near zero" (e.g. a component that is
        # 0 by symmetry, where a relative diff is meaningless) from a real small value
        field_scale = max(np.abs(v1).max(), np.abs(v2).max(), 1e-300)
        threshold = tol + rtol * np.abs(v2)

        if np.any(abs_diff > threshold):
            ind = np.where(abs_diff > threshold)[0]
            centers = cells1[index1[ind]].mean(axis=1)
            worst = np.argmax(abs_diff[ind])
            print(
                f"{field} is not the same between {file1} and {file2}: "
                f"{len(ind)}/{len(v1)} cells differ  "
                f"max|abs diff|={abs_diff[ind].max():.6e} (tol={tol:.3e})  "
                f"max|rel diff|={rel_diff[ind].max():.6e} (rtol={rtol:.3e})  "
                f"worst at center~{centers[worst]}"
            )
            if args.verbose:
                print(
                    f"{'idx':>8s}  {'center (x,y,z)':>30s}  "
                    f"{'value 1':>22s}  {'value 2':>22s}  "
                    f"{'abs diff':>12s}  {'rel diff':>12s}  note"
                )
                for i, c in zip(ind, centers):
                    # a relative diff can look huge on a value that is itself ~0
                    # (e.g. a component that is 0 by symmetry): flag it so it isn't
                    # mistaken for a large real discrepancy
                    note = (
                        "value~0 rel. to field scale, rel diff not meaningful"
                        if denom[i] < 1e-6 * field_scale
                        else ""
                    )
                    print(
                        f"{i:8d}  ({c[0]: .8f},{c[1]: .8f},{c[2]: .8f})  "
                        f"{v1[i]: .15e}  {v2[i]: .15e}  "
                        f"{abs_diff[i]:.6e}  {rel_diff[i]:.6e}  {note}"
                    )
            check = False
    if not check:
        sys.exit(f"files {file1} and {file2} are different")

    print(f"files {file1} and {file2} are the same")


if args.start is not None and args.end is not None:
    for i in range(args.start, args.end + 1):
        compare_meshes(
            f"{args.file1}{i}.h5", f"{args.file2}{i}.h5", args.tol, args.rtol
        )
else:
    compare_meshes(f"{args.file1}.h5", f"{args.file2}.h5", args.tol, args.rtol)
