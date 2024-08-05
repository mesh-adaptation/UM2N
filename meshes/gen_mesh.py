import os
import numpy as np

airfoil_name = "naca4415.dat"

in_file = os.path.join(os.path.dirname(__file__), "airfoil", airfoil_name)


def genMesh(airfoilFile):
    scale_factor = 4 ## scale the points by this scalar: Coord(point) / scale
    lc = 0.015
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, {}}};\n".format(pointIndex, ar[n][0]/scale_factor, ar[n][1]/scale_factor, lc)
        pointIndex += 1

    with open(os.path.join(os.path.dirname(__file__), "airfoil", "airfoil_template.geo"), "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                outFile.write(line)

    # if os.system("gmsh airfoil.geo -2 -o airfoil.msh -format msh2 > /dev/null") != 0:
    if os.system("gmsh airfoil.geo -2 -o airfoil.msh -format msh2") != 0:
        print("error during mesh creation!")
        return(-1)

    # if os.system("gmshToFoam airfoil.msh > /dev/null") != 0:
    #     print("error during conversion to OpenFoam mesh!")
    #     return(-1)

    # with open(os.path.join(
    #     os.path.dirname(os.path.dirname(__file__)),
    #     "constant", "polyMesh", "boundary"), "rt") as inFile:
    #     with open(os.path.join(
    #         os.path.dirname(os.path.dirname(__file__)),
    #         "constant", "polyMesh", "boundaryTemp"), "wt") as outFile:
    #         inBlock = False
    #         inAerofoil = False
    #         for line in inFile:
    #             if "front" in line or "back" in line:
    #                 inBlock = True
    #             elif "aerofoil" in line:
    #                 inAerofoil = True
    #             if inBlock and "type" in line:
    #                 line = line.replace("patch", "empty")
    #                 inBlock = False
    #             if inAerofoil and "type" in line:
    #                 line = line.replace("patch", "wall")
    #                 inAerofoil = False
    #             outFile.write(line)
    # os.rename("constant/polyMesh/boundaryTemp","constant/polyMesh/boundary")

    return(0)

if __name__ == "__main__":
    genMesh(in_file)
