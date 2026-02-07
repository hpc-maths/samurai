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
    default=1e-14,
    help="tolerance for field comparison",
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


def compare_meshes(file1, file2, tol):
    mesh1 = h5py.File(file1, "r")["mesh"]
    mesh2 = h5py.File(file2, "r")["mesh"]
    cells1 = construct_cells(mesh1)
    cells2 = construct_cells(mesh2)

    index1 = np.argsort(np.asarray([c.tobytes() for c in cells1]))
    index2 = np.argsort(np.asarray([c.tobytes() for c in cells2]))

    if np.any(cells1.shape != cells2.shape):
        print("shape are not compatible")
        print(f"{cells1.shape} vs {cells2.shape}")
        sys.exit(f"files {file1} and {file2} are different")

    if np.any(cells1[index1] != cells2[index2]):
        print("cells are not the same")
        sys.exit(f"files {file1} and {file2} are different")

    field1 = construct_fields(mesh1)
    field2 = construct_fields(mesh2)

    for field in field1.keys():
        if not field in field2.keys():
            print(f"{field} is not in second file")
            sys.exit(f"files {file1} and {file2} are different")

        tol = rtol * np.abs(field1[field][:][index1])  # relative tolerance
        # tol[np.where(np.abs(field1[field][:][index1]) < eps)] = rtol # where the field is close to machine zero, switch to absolute tolerance
        tol[np.abs(field1[field][:][index1] - field2[field][:][index2]) < eps] = (
            np.inf
        )  # where the field is close to machine zero, switch to absolute tolerance
        ind = np.where(
            np.abs(field1[field][:][index1] - field2[field][:][index2]) > tol
        )
        if len(ind[0]) > 0:
            idx = np.argmax(
                np.abs(field1[field][:][index1][ind] - field2[field][:][index2][ind])
            )
            # ind = np.where(field1[field][:][index1] != field2[field][:][index2])
            print(f"-- Values {file1}")
            print(field1[field][:][index1[ind]])
            print(f"-- Values {file2}")
            print(field2[field][:][index2[ind]])
            print("-- Difference:")
            print(np.abs(field1[field][:][index1[ind]] - field2[field][:][index2[ind]]))
            print("-- Coordinates (x,y,z):")
            print(cells1[index1[ind]], cells2[index2[ind]])
            # print(np.abs(field1[field][:][index1[ind]]-field2[field][:][index2[ind]]))
            sys.exit(f"{field} is not the same between {file1} and {file2}")
    print(f"files {file1} and {file2} are the same")


if args.start is not None and args.end is not None:
    for i in range(args.start, args.end + 1):
        compare_meshes(f"{args.file1}{i}.h5", f"{args.file2}{i}.h5", args.tol)
else:
    compare_meshes(f"{args.file1}.h5", f"{args.file2}.h5", args.tol)
