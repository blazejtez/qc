#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, jit, prange


@jit(nopython=True, parallel=True)
def _evaluate(x_linspace: np.ndarray, y_linspace: np.ndarray,
              z_linspace: np.ndarray, Z1: float, Z2: float) -> np.ndarray:

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
                    z] = Z1 * Z2 * 1 / np.sqrt(xval**2 + yval**2 + zval**2)

    return out


class Potential:
    """Potential. Computes Coulomb potential for a 3D mesh"""
    def __init__(self, Z1: float = 1, Z2: float = 1):
        self.Z1 = Z1
        self.Z2 = Z2

    def evaluate(self, xl, yl, zl):

        return _evaluate(xl, yl, zl, self.Z1, self.Z2)


if __name__ == "__main__":

    xl = np.linspace(-1, 1, 2501, dtype=np.float32)
    yl = np.linspace(-1, 1, 2501,  dtype=np.float32)
    zl = np.linspace(-1, 1, 2501, dtype=np.float32)
    p = Potential()
    print('start computation of potential ...')
    out = p.evaluate(xl, yl, zl)
