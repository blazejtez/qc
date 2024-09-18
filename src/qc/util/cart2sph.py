#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numexpr as ne
import numpy as np
from qc.data import raster
import portion as P


class Cart2Sph:
    def __init__(self, x_linspace, y_linspace, z_linspace):

        self.x, self.y, self.z = np.meshgrid(x_linspace, y_linspace,
                                             z_linspace)

    def _cart2sph(self, x, y, z, ceval=ne.evaluate):
        """ x, y, z :  ndarray coordinates
            ceval: backend to use:
                  - eval :  pure Numpy
                  - numexpr.evaluate:  Numexpr """
        azimuth = ceval('arctan2(y,x)')
        print(f"{np.min(azimuth)}, {np.max(azimuth)}")
        xy2 = ceval('x**2 + y**2')
        elevation = ceval('arctan2(sqrt(xy2),z)')
        print(f"{np.min(elevation)}, {np.max(elevation)}")
        r = ceval('sqrt(xy2 + z**2)')
        return r, azimuth, elevation

    def evaluate(self):

        r, azimuth, elevation = self._cart2sph(self.x,self.y,self.z)
        print(azimuth.min())
        print(elevation.min())
        print(r.min())

        return r, azimuth, elevation, self.x, self.y, self.z


if __name__ == "__main__":

    r = raster.Raster(1)
    x_linspace = r.linspace(P.closed(-1., 1))

    c = Cart2Sph(x_linspace, x_linspace, x_linspace)

    print(c.evaluate())
