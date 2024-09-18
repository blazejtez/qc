#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from typing import Tuple

from qc.util import contour
import numpy as np
import portion as P
from mayavi.mlab import contour3d, show

import Praktyki.cut_box_3D as box
import qc.psi.hydrogenpsi as hydrogenpsi
import qc.data.raster as raster


def analytical(quantum_numbers: Tuple, box: box.box3D, raster_density=50):
    """analytical.

    :param quantum_numbers: (n,l,m) - principal, orbital and magnetic quantum numbers
    :type quantum_numbers: Tuple
    :param box: 3D box containing the wave function
    :type box: box.box3D
    :param raster_density: number of grid points per Bohr radius
    """
    r = raster.Raster(raster_density)
    x_linspace, y_linspace, z_linspace = r.box_linspaces(box)
    x, y, z = np.meshgrid(x_linspace, y_linspace, z_linspace)
    psi = hydrogenpsi.HydrogenPsi(*quantum_numbers)
    psi_data = psi.evaluate(x, y, z)
    return psi_data, x, y, z


def numerical(quantum_numbers: Tuple, box: box.box3D, raster_density=50):
    """numerical.

    :param quantum_numbers: (n,l,m) - principal, orbital and magnetic quantum numbers
    :type quantum_numbers: Tuple
    :param box: 3D box containing the wave function
    :type box: box.box3D
    :param raster_density: number of grid points per Bohr radius
    """
    r = raster.Raster(raster_density)
    x_linspace, y_linspace, z_linspace = r.box_linspaces(box)
    x, y, z = np.meshgrid(x_linspace, y_linspace, z_linspace)
    psi = hydrogenpsi.HydrogenPsi(*quantum_numbers)
    psi_data = psi.evaluate(x, y, z)
    return psi_data, x, y, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n",
        "--numerical",
        action="store_true",
        help="compute hydrogen wave functions and energy numerically")
    group.add_argument(
        "-a",
        "--analytical",
        action="store_true",
        help="compute hydrogen wave functions and energy analytically")
    parser.add_argument(
        "principal",
        type=int,
        choices=[1, 2, 3, 4],
        help="principal quantum number",
    )
    # parser.add_argument(
    #    "folder",
    #    help=
    #    "directory to save the 3D drawings of the modules squared of the wavefunctions"
    # )
    args = parser.parse_args()

    n = args.principal
    # plot wavefunction computed analytically
    HYDROGEN_RADIUS = 35.
    ISOLINE_LEVEL = 80.  # in percent of the total mass
    for l in range(0, n):
        for m in range(0, l + 1):
            if args.analytical:
                print("Calculating analytical wave functions")
                print(f"Quantum numbers: principal {n}, orbital {l}, magnetic {m}")
                print("Wait!!!")
                intvl = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
                box_ = box.box3D(intvl, intvl, intvl)
                psi, x, y, z = analytical((n, l, m), box_, raster_density=5)
                c = contour.Contour(psi)
                v = c.get_isoline(ISOLINE_LEVEL)
                contour3d(psi, contours=[v])
                show()
            elif args.numerical:
                print("Calculating numerical wave functions")
                print(f"Quantum numbers: principal {n}, orbital {l}, magnetic {m}")
                print("Wait!!!")
                intvl = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
                box_ = box.box3D(intvl, intvl, intvl)
                psi, x, y, z = numerical((n, l, m), box_, raster_density=5)
                c = contour.Contour(psi)
                v = c.get_isoline(ISOLINE_LEVEL)
                contour3d(psi, contours=[v])
                show()
