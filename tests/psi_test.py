#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from typing import Tuple

import numpy as np
import portion as P

import Praktyki.cut_box as box
import qc.hydrogenpsi as hydrogenpsi
import qc.raster as raster


class MainTest(unittest.TestCase):
    def test(self):

        n, l, m = 2, 1, 1

        HYDROGEN_RADIUS = 35.

        intvl = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
        box_ = box.box3D(intvl, intvl, intvl)

        psi, psi_re, psi_im, x, y, z = hydrogenpsi.analytical((n, l, m),
                                                  box_,
                                                  raster_density=5)

        self.assertAlmostEqual(np.sum((psi - psi_re**2 - psi_im**2)**2), 0.0)


if __name__ == "__main__":

    unittest.main()
