#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numexpr as ne
import numpy as np
import raster
import portion as P
class Cart2Sph:

    def __init__(self, x_linspace, y_linspace, z_linspace):

        self.x,self.y,self.z = np.meshgrid(x_linspace,y_linspace,z_linspace)
                

    def _cart2sph(self, x, y, z, ceval=ne.evaluate):
        """ x, y, z :  ndarray coordinates
            ceval: backend to use:
                  - eval :  pure Numpy
                  - numexpr.evaluate:  Numexpr """
        azimuth = ceval('arctan2(y,x)')
        xy2 = ceval('x**2 + y**2')
        elevation = ceval('arctan2(z, sqrt(xy2))')
        r = ceval('sqrt(xy2 + z**2)')
        return azimuth, elevation, r

    def evaluate(self):

        return self._cart2sph(self.x,self.y,self.z)

if __name__ == "__main__":

    r = raster.Raster(1)
    x_linspace = r.linspace(P.closed(-1.,1))
    
    c = Cart2Sph(x_linspace,x_linspace,x_linspace)

    print(c.evaluate())
