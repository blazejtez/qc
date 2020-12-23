#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module responsible for the computation of the laplacian of a cube'''

import numpy as np
from typing import Dict
from numba import jit, prange
import matplotlib.pyplot as plt


@jit(nopython=True, parallel=True)
def _eval_laplacian3d_7pts_stencil(cube: np.ndarray, xlen: int, ylen: int,
                                   zlen: int) -> np.ndarray:
    cube_out = np.empty_like(cube)

    for x in prange(1, xlen - 1):
        for y in prange(1, ylen - 1):
            for z in prange(1, zlen - 1):
                cube_out[x, y, z] = -6 * cube[x, y, z]
                cube_out[x, y, z] += cube[x - 1, y, z]
                cube_out[x, y, z] += cube[x + 1, y, z]
                cube_out[x, y, z] += cube[x, y - 1, z]
                cube_out[x, y, z] += cube[x, y + 1, z]
                cube_out[x, y, z] += cube[x, y, z - 1]
                cube_out[x, y, z] += cube[x, y, z + 1]

    for x in prange(1, xlen - 1):
        for y in prange(1, ylen - 1):
            cube_out[x, y, 0] = -6 * cube[x, y, 0]
            cube_out[x, y, zlen - 1] = -6 * cube[x, y, zlen - 1]
            cube_out[x, y, 0] += cube[x - 1, y, 0]
            cube_out[x, y, zlen - 1] += cube[x - 1, y, zlen - 1]
            cube_out[x, y, 0] += cube[x + 1, y, 0]
            cube_out[x, y, zlen - 1] += cube[x + 1, y, zlen - 1]
            cube_out[x, y, 0] += cube[x, y - 1, 0]
            cube_out[x, y, zlen - 1] += cube[x, y - 1, zlen - 1]
            cube_out[x, y, 0] += cube[x, y + 1, 0]
            cube_out[x, y, zlen - 1] += cube[x, y + 1, zlen - 1]
            cube_out[x, y, 0] += cube[x, y, 1]
            cube_out[x, y, zlen - 1] += cube[x, y, zlen - 2]

    for x in prange(1, xlen - 1):
        for z in prange(1, zlen - 1):
            cube_out[x, 0, z] = -6 * cube[x, 0, z]
            cube_out[x, ylen - 1, z] = -6 * cube[x, ylen - 1, z]

            cube_out[x, 0, z] += cube[x, 0, z - 1]
            cube_out[x, ylen - 1, z] += cube[x, ylen - 1, z - 1]

            cube_out[x, 0, z] += cube[x, 0, z + 1]
            cube_out[x, ylen - 1, z] += cube[x, ylen - 1, z + 1]

            cube_out[x, 0, z] += cube[x - 1, 0, z]
            cube_out[x, ylen - 1, z] += cube[x - 1, ylen - 1, z]

            cube_out[x, 0, z] += cube[x + 1, 0, z]
            cube_out[x, ylen - 1, z] += cube[x + 1, ylen - 1, z]

            cube_out[x, 0, z] += cube[x, 1, z]
            cube_out[x, ylen - 1, z] += cube[x, ylen - 2, z]

    for y in prange(1, ylen - 1):
        for z in prange(1, zlen - 1):

            cube_out[0, y, z] = -6 * cube[0, y, z]
            cube_out[xlen - 1, y, z] = -6 * cube[xlen - 1, y, z]

            cube_out[0, y, z] += cube[0, y - 1, z]
            cube_out[xlen - 1, y, z] += cube[xlen - 1, y - 1, z]

            cube_out[0, y, z] += cube[0, y + 1, z]
            cube_out[xlen - 1, y, z] += cube[xlen - 1, y + 1, z]

            cube_out[0, y, z] += cube[0, y, z - 1]
            cube_out[xlen - 1, y, z] += cube[xlen - 1, y, z - 1]

            cube_out[0, y, z] += cube[0, y, z + 1]
            cube_out[xlen - 1, y, z] += cube[xlen - 1, y, z + 1]

            cube_out[0, y, z] += cube[1, y, z]
            cube_out[xlen - 1, y, z] += cube[xlen - 2, y, z]

    for x in prange(1, xlen - 1):
        cube_out[x, 0, 0] = -6 * cube[x, 0, 0]
        cube_out[x, ylen - 1, 0] = -6 * cube[x, ylen - 1, 0]
        cube_out[x, 0, zlen - 1] = -6 * cube[x, 0, zlen - 1]
        cube_out[x, ylen - 1, zlen - 1] = -6 * cube[x, ylen - 1, zlen - 1]

        cube_out[x, 0, 0] += cube[x - 1, 0, 0]
        cube_out[x, ylen - 1, 0] += cube[x - 1, ylen - 1, 0]
        cube_out[x, 0, zlen - 1] += cube[x - 1, 0, zlen - 1]
        cube_out[x, ylen - 1, zlen - 1] += cube[x - 1, ylen - 1, zlen - 1]

        cube_out[x, 0, 0] += cube[x + 1, 0, 0]
        cube_out[x, ylen - 1, 0] += cube[x + 1, ylen - 1, 0]
        cube_out[x, 0, zlen - 1] += cube[x + 1, 0, zlen - 1]
        cube_out[x, ylen - 1, zlen - 1] += cube[x + 1, ylen - 1, zlen - 1]

        cube_out[x, 0, 0] += cube[x, 1, 0]
        cube_out[x, 0, 0] += cube[x, 0, 1]
        cube_out[x, ylen - 1, 0] += cube[x, ylen - 2, 0]
        cube_out[x, ylen - 1, 0] += cube[x, ylen - 1, 1]
        cube_out[x, 0, zlen - 1] += cube[x, 1, zlen - 1]
        cube_out[x, 0, zlen - 1] += cube[x, 0, zlen - 2]
        cube_out[x, ylen - 1, zlen - 1] += cube[x, ylen - 2, zlen - 1]
        cube_out[x, ylen - 1, zlen - 1] += cube[x, ylen - 1, zlen - 2]

    for y in prange(1, ylen - 1):
        cube_out[0, y, 0] = -6 * cube[0, y, 0]
        cube_out[xlen - 1, y, 0] = -6 * cube[xlen - 1, y, 0]
        cube_out[0, y, zlen - 1] = -6 * cube[0, y, zlen - 1]
        cube_out[xlen - 1, y, zlen - 1] = -6 * cube[xlen - 1, y, zlen - 1]

        cube_out[0, y, 0] += cube[0, y - 1, 0]
        cube_out[xlen - 1, y, 0] += cube[xlen - 1, y - 1, 0]
        cube_out[0, y, zlen - 1] += cube[0, y - 1, zlen - 1]
        cube_out[xlen - 1, y, zlen - 1] += cube[xlen - 1, y - 1, zlen - 1]

        cube_out[0, y, 0] += cube[0, y + 1, 0]
        cube_out[xlen - 1, y, 0] += cube[xlen - 1, y + 1, 0]
        cube_out[0, y, zlen - 1] += cube[0, y + 1, zlen - 1]
        cube_out[xlen - 1, y, zlen - 1] += cube[xlen - 1, y + 1, zlen - 1]

        cube_out[0, y, 0] += cube[1, y, 0]
        cube_out[0, y, 0] += cube[0, y, 1]
        cube_out[xlen - 1, y, 0] += cube[xlen - 2, y, 0]
        cube_out[xlen - 1, y, 0] += cube[xlen - 1, y, 1]
        cube_out[0, y, zlen - 1] += cube[1, y, zlen - 1]
        cube_out[0, y, zlen - 1] += cube[0, y, zlen - 2]
        cube_out[xlen - 1, y, zlen - 1] += cube[xlen - 2, y, zlen - 1]
        cube_out[xlen - 1, y, zlen - 1] += cube[xlen - 1, y, zlen - 2]

    for z in prange(1, zlen - 1):
        cube_out[0, 0, z] = -6 * cube[0, 0, z]
        cube_out[xlen - 1, 0, z] = -6 * cube[xlen - 1, 0, z]
        cube_out[0, ylen - 1, z] = -6 * cube[0, ylen - 1, z]
        cube_out[xlen - 1, ylen - 1, z] = -6 * cube[xlen - 1, ylen - 1, z]

        cube_out[0, 0, z] += cube[0, 0, z - 1]
        cube_out[xlen - 1, 0, z] += cube[xlen - 1, 0, z - 1]
        cube_out[0, ylen - 1, z] += cube[0, ylen - 1, z - 1]
        cube_out[xlen - 1, ylen - 1, z] += cube[xlen - 1, ylen - 1, z - 1]

        cube_out[0, 0, z] += cube[0, 0, z + 1]
        cube_out[xlen - 1, 0, z] += cube[xlen - 1, 0, z + 1]
        cube_out[0, ylen - 1, z] += cube[0, ylen - 1, z + 1]
        cube_out[xlen - 1, ylen - 1, z] += cube[xlen - 1, ylen - 1, z + 1]

        cube_out[0, 0, z] += cube[1, 0, z]
        cube_out[0, 0, z] += cube[0, 1, z]
        cube_out[xlen - 1, 0, z] += cube[xlen - 2, 0, z]
        cube_out[xlen - 1, 0, z] += cube[xlen - 1, 1, z]
        cube_out[0, ylen - 1, z] += cube[1, ylen - 1, z]
        cube_out[0, ylen - 1, z] += cube[0, ylen - 2, z]
        cube_out[xlen - 1, ylen - 1, z] += cube[xlen - 2, ylen - 1, z]
        cube_out[xlen - 1, ylen - 1, z] += cube[xlen - 1, ylen - 2, z]

    cube_out[0, 0, 0] = -6 * cube[0, 0, 0]
    cube_out[0, 0, 0] += cube[1, 0, 0] + cube[0, 1, 0] + cube[0, 0, 1]
    cube_out[xlen - 1, 0, 0] = -6 * cube[xlen - 1, 0, 0]
    cube_out[xlen - 1, 0,
             0] += cube[xlen - 2, 0, 0] + cube[xlen - 1, 1, 0] + cube[xlen - 1,
                                                                      0, 1]

    cube_out[xlen - 1, ylen - 1, 0] = -6 * cube[xlen - 1, ylen - 1, 0]
    cube_out[xlen - 1, ylen - 1,
             0] += cube[xlen - 2, ylen - 1, 0] + cube[xlen - 1, ylen - 2,
                                                      0] + cube[xlen - 1,
                                                                ylen - 1, 1]
    cube_out[xlen - 1, 0, zlen - 1] = -6 * cube[xlen - 1, 0, zlen - 1]
    cube_out[xlen - 1, 0, zlen -
             1] += cube[xlen - 2, 0, zlen - 1] + cube[xlen - 1, 1, zlen -
                                                      1] + cube[xlen - 1, 0,
                                                                zlen - 2]

    cube_out[0, ylen - 1, 0] = -6 * cube[0, ylen - 1, 0]

    cube_out[0, ylen - 1,
             0] += cube[1, ylen - 1, 0] + cube[0, ylen - 2,
                                               0] + cube[0, ylen - 1, 1]

    cube_out[0, ylen - 1, zlen - 1] = -6 * cube[0, ylen - 1, zlen - 1]

    cube_out[0, ylen - 1, zlen -
             1] += cube[1, ylen - 1, zlen - 1] + cube[0, ylen - 2, zlen -
                                                      1] + cube[0, ylen - 1,
                                                                zlen - 2]

    cube_out[0, 0, zlen - 1] = -6 * cube[0, 0, zlen - 1]

    cube_out[0, 0, zlen -
             1] += cube[1, 0, zlen - 1] + cube[0, 1, zlen - 1] + cube[0, 0,
                                                                      zlen - 2]

    cube_out[xlen-1,ylen-1,zlen-1] = -6*cube[xlen-1,ylen-1,zlen-1]
    cube_out[xlen-1,ylen-1,zlen-1] += cube[xlen-2,ylen-1,zlen-1] + cube[xlen-1,ylen-2,zlen-1] + cube[xlen-1,ylen-1,zlen-2]


    return cube_out


class Stencils3D:
    """Stencils3D."""
    def __init__(self):

        self.stencil2 = {
            (0, 0, 0): -6,
            (-1, 0, 0): 1,
            (1, 0, 0): 1,
            (0, -1, 0): 1,
            (0, 1, 0): 1,
            (0, 0, -1): 1,
            (0, 0, 1): 1
        }


class Laplacian3D:
    def __init__(self, shape: tuple, stencil: Dict[tuple, int], h: float = 1.):
        self.shape = shape
        self.shape_np = np.asarray(self.shape)
        self.stencil = stencil
        self.h3 = h**3

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        vec = np.reshape(vec, self.shape)
        vec_out = self.matcube(vec)
        return np.reshape(vec_out, (np.prod(self.shape_np), 1))

    def matcube(self, cube: np.ndarray) -> np.ndarray:

        cube_out = np.zeros(self.shape, dtype=np.float32)
        for k, v in self.stencil.items():
            k_np = np.asarray(k)
            lv = self._lower_vec(k_np)
            uv = self._upper_vec(k_np)
            lvo = self._lower_vec_out(k_np)
            uvo = self._upper_vec_out(k_np)
            cube_out[lvo[0]:uvo[0], lvo[1]:uvo[1],
                     lvo[2]:uvo[2]] += v * cube[lv[0]:uv[0], lv[1]:uv[1],
                                                lv[2]:uv[2]] * self.h3

        return cube_out

    def matcube_numba(self, cube: np.ndarray) -> np.ndarray:
        xlen = np.size(cube, 0)
        ylen = np.size(cube, 1)
        zlen = np.size(cube, 2)
        return _eval_laplacian3d_7pts_stencil(cube, xlen, ylen, zlen)

    def _lower_vec(self, k: np.ndarray):
        return np.maximum(k, np.array([0, 0, 0]))

    def _upper_vec(self, k: np.ndarray):
        return np.minimum(self.shape + k, self.shape)

    def _lower_vec_out(self, k: np.ndarray):
        return np.maximum(-k, np.array([0, 0, 0]))

    def _upper_vec_out(self, k: np.ndarray):
        return np.minimum(self.shape - k, self.shape)


if __name__ == "__main__":

    st = Stencils3D()

    shape = (4, 4, 4)

    lap = Laplacian3D(shape, st.stencil2)

    cube = np.random.randn(*shape).astype(np.float32)

    cube_out = lap.matcube(cube)

    cube_out_numba = lap.matcube_numba(cube)
    
    print((cube_out-cube_out_numba)**2)

    plt.plot(cube_out.flatten())
    plt.plot(cube_out_numba.flatten())
    plt.show()
