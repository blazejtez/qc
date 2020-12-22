#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import laplacian3Dnumpy
import potentialnumpy


class HydrogenHamiltonian:

    def __init__(self,x_linspace: np.ndarray, y_linspace: np.ndarray, z_linspace: np.ndarray, h: float = 1 ):

        p = potentialnumpy.Potential(1,1)

        self.pcube = p.evaluate(x_linspace, y_linspace, z_linspace)

        self.shape = (len(x_linspace), len(y_linspace), len(z_linspace))

        s = laplacian3Dnumpy.Stencils3D()

        self.l = laplacian3Dnumpy.Laplacian3D(self.shape, s.stencil2, h = h)

    def operate(self,cube :np.ndarray) -> np.ndarray:
        
        cube *= self.pcube # apply potential

        cube += .5*self.l.matcube(cube)

        return cube
    

if __name__ == "__main__":

    xl = np.linspace(-30,30,1001, dtype = np.float32)
    yl = xl
    zl = xl
    
    h = HydrogenHamiltonian(xl,yl,zl,h = xl[1]-xl[0])

    cube = np.random.randn(1001,1001,1001)

    cube = h.operate(cube)


