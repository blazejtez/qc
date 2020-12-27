#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from qc.laplacian3d import *
from qc.potential import *
import time
from numba import jit, prange


@jit(nopython=True, parallel=True)
def _add_cubes_hamiltonian(c1laplacian: np.ndarray, c2potential: np.ndarray,
                           xlen: int, ylen: int, zlen: int):
    for x in prange(xlen):
        for y in prange(ylen):
            for z in prange(zlen):
                c2potential[x, y, z] -= .5 * c1laplacian[x, y, z]

    return c2potential


class HydrogenHamiltonian:
    def __init__(self, x_linspace: np.ndarray, y_linspace: np.ndarray,
                 z_linspace: np.ndarray):
        """__init__.

        :param x_linspace:
        :type x_linspace: np.ndarray
        :param y_linspace:
        :type y_linspace: np.ndarray
        :param z_linspace:
        :type z_linspace: np.ndarray
        """

        self.p = Potential()
        self.xl = x_linspace
        self.yl = y_linspace
        self.zl = z_linspace

        self.shape = (len(x_linspace), len(y_linspace), len(z_linspace))

        hx = self.xl[1] - self.xl[0]
        hy = self.yl[1] - self.yl[0]
        hz = self.zl[1] - self.zl[0]

        # check if the grid cell is cubic
        assert (abs(hy - hx) < 1e-7)
        assert (abs(hz - hx) < 1e-7)
        assert (abs(hz - hy) < 1e-7)

        self.l = Laplacian3D(h=hx)

    def operate(self, cube: np.ndarray) -> np.ndarray:

        cube_l = self.l.matcube_numba(cube)

        cube = self.p.operate(cube, self.xl, self.yl, self.zl)

        xlen = np.size(cube, 0)
        ylen = np.size(cube, 1)
        zlen = np.size(cube, 2)

        return _add_cubes_hamiltonian(cube_l, cube, xlen, ylen, zlen)

    def operate_vec(self, vecs: np.ndarray) -> np.ndarray:

        for i in range(np.size(vecs, 1)):

            tic = time.time()
            cube = np.reshape(vecs[:, i], self.shape)
            toc = time.time()
            print(f"time elapsed reshape: {toc-tic} sec.")
            tic = time.time()
            cube = self.operate(cube)
            toc = time.time()
            print(f"time elapsed operate: {toc-tic} sec.")
            tic = time.time()
            vecs[:, i] = np.reshape(cube, (np.prod(self.shape), ))
            toc = time.time()
            print(f"time elapsed reshape: {toc-tic} sec.")

        return vecs


if __name__ == "__main__":

    ln = 1001

    xl = np.linspace(-30, 30, ln, dtype=np.float32)
    yl = xl
    zl = xl

    h = HydrogenHamiltonian(xl, yl, zl)
    cube = np.random.randn(ln, ln, ln).astype(np.float32)
    print("start applying Hamiltonian ...")
    tic = time.time()
    cube = h.operate(cube)
    toc = time.time()
    print(f"time elapsed: {toc-tic} sec.")

    print("start applying Hamiltonian ...")
    tic = time.time()
    cube = h.operate(cube)
    toc = time.time()
    print(f"time elapsed: {toc-tic} sec.")

    vecs = np.random.randn(np.prod((ln, ln, ln)), 3).astype(np.float32)
    print("start applying Hamiltonian ...")
    tic = time.time()
    vecs = h.operate_vec(vecs)
    toc = time.time()
    print(f"time elapsed: {toc-tic} sec.")

    vecs = np.random.randn(np.prod((ln, ln, ln)), 3).astype(np.float32)
    print("start applying Hamiltonian ...")
    tic = time.time()
    vecs = h.operate_vec(vecs)
    toc = time.time()
    print(f"time elapsed: {toc-tic} sec.")
