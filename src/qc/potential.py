#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from numba import njit, jit, prange


@jit(nopython=True, parallel=True)
def _evaluate(x_linspace: np.ndarray, y_linspace: np.ndarray,
              z_linspace: np.ndarray, Z1: float, Z2: float, eps: float = 1e-8) -> np.ndarray:

    xlen = len(x_linspace)
    ylen = len(y_linspace)
    zlen = len(z_linspace)

    out = np.empty((xlen, ylen, zlen))

    for x in prange(xlen):
        xval = x_linspace[x]
        for y in prange(ylen):
            yval = y_linspace[y]
            for z in prange(zlen):
                zval = z_linspace[z]
                out[x, y,
                    z] = Z1 * Z2 * 1 / (np.sqrt(xval**2 + yval**2 + zval**2 )+eps)

    return out


@jit(nopython=True, parallel=True)
def _operate(cube: np.ndarray, x_linspace: np.ndarray, y_linspace: np.ndarray,
              z_linspace: np.ndarray, Z1: float, Z2: float, eps: float = 1e-8) -> np.ndarray:

    xlen = len(x_linspace)
    ylen = len(y_linspace)
    zlen = len(z_linspace)

    for x in prange(xlen):
        xval = x_linspace[x]
        x2 = xval**2
        for y in prange(ylen):
            yval = y_linspace[y]
            x2y2 = x2 + yval**2
            for z in prange(zlen):
                zval = z_linspace[z]
                x2y2z2 = x2y2+zval**2
                cube[x, y,
                    z] *= Z1 * Z2 * 1 / max(np.sqrt(x2y2z2),eps) 

    return cube
class Potential:
    """Potential. Computes Coulomb potential for a 3D mesh"""
    def __init__(self, Q1: float = 1, Q2: float = 1):
        """__init__.

        :param Q1: charge of the first particle in atomic units
        :type Q1: float
        :param Q2: charge of the second particla in atomic units
        :type Q2: float
        """
        self.Q1 = Q1
        self.Q2 = Q2

    def evaluate(self, xl, yl, zl):

        return _evaluate(xl, yl, zl, self.Q1, self.Q2)

    def operate(self, cube, xl, yl, zl):
        assert(np.size(cube,0) == len(xl))
        assert(np.size(cube,1) == len(yl))
        assert(np.size(cube,2) == len(zl))

        return _operate(cube, xl,yl, zl, self.Q1, self.Q2)


if __name__ == "__main__":

    xl = np.linspace(-1, 1, 1001, dtype=np.float32)
    yl = np.linspace(-1, 1, 1001,  dtype=np.float32)
    zl = np.linspace(-1, 1, 1001, dtype=np.float32)
    print('start generating random cube ...')
    tic = time.time()
    cube = np.random.randn(len(xl),len(yl),len(zl)).astype(np.float32)
    toc = time.time()
    print(f"time elapsed: {toc-tic} sec.")
    p = Potential()
    print('start computation of potential ...')
    tic = time.time()
    out = p.operate(cube, xl, yl, zl)
    toc = time.time()
    print(f'time elapsed: {toc-tic} sec.')

    print('start computation of potential ...')
    tic = time.time()
    out = p.operate(cube, xl, yl, zl)
    toc = time.time()
    print(f'time elapsed: {toc-tic} sec.')
