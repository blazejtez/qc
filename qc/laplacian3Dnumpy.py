#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict


class Stencils3D:
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
                     lvo[2]:uvo[2]] += v *cube[lv[0]:uv[0], lv[1]:uv[1],
                                               lv[2]:uv[2]]*self.h3

        return cube_out

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

    shape = (100, 100, 100)

    lap = Laplacian3D(shape, st.stencil2)

    vec = np.random.randn(*shape).astype(np.float32)

    vec_out = lap.matvec(vec)

    print(vec_out)
