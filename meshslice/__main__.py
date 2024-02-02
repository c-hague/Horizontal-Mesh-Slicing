import argparse
import pyvista as pv
from .slicer import polygonsFromMesh
from pprint import pprint
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog='meshslice',
        description="Slices a mesh along a horizontal plane and creates polygons from the intersections."
    )
    parser.add_argument('--file', type=str, required=True,
                        help="path to mesh file")
    parser.add_argument('-z', type=float, help="z height float")
    args = parser.parse_args()

    mesh = pv.read(args.file)
    polygons = polygonsFromMesh(args.z, mesh)

    asArrays = []
    for polygon in polygons:
        asArrays.append(np.array(polygon.exterior.coords.xy).T.tolist())

    pprint(asArrays)


if __name__ == '__main__':
    main()
