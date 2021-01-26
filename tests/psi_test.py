#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from typing import Tuple

import numpy as np
import portion as P

import Praktyki.cut_box as box
import qc.hydrogenpsi as hydrogenpsi
import qc.raster as raster


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
    psi_data_re, psi_data_im = psi.evaluate_complex(x, y, z)
    psi_data = psi.evaluate(x, y, z)
    return psi_data, psi_data_re, psi_data_im, x, y, z


class MainTest(unittest.TestCase):
    def test(self):

        n, l, m = 2, 1, 1

        HYDROGEN_RADIUS = 1.

        intvl = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
        box_ = box.box3D(intvl, intvl, intvl)

        psi, psi_re, psi_im, x, y, z = analytical((n, l, m),
                                                  box_,
                                                  raster_density=1)

        print(np.any(np.isnan(psi)))
        print(np.any(np.isnan(psi_re)))
        print(np.any(np.isnan(psi_im)))
        print(psi_re)
        print(psi_im)

        self.assertAlmostEqual(np.sum((psi - psi_re**2 - psi_im**2)**2), 0.0)


if __name__ == "__main__":

    unittest.main()
