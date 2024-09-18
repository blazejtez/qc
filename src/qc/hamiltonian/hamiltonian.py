#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import numpy as np

from qc.hamiltonian.potential import *
from qc.hamiltonian.laplacian3d import Laplacian3D
from qc.hamiltonian.potential import Potential
import qc.data.surface as S
import qc.data.texture3D as T

class HydrogenHamiltonian:
    BLOCKSIZE = 8
    eps = 1e-4
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

        self.xl = x_linspace
        self.yl = y_linspace
        self.zl = z_linspace

        self.shape = (len(x_linspace), len(y_linspace), len(z_linspace))

        hx = self.xl[1] - self.xl[0]
        hy = self.yl[1] - self.yl[0]
        hz = self.zl[1] - self.zl[0]
        self.h = hx
        # check if the grid cell is cubic
        assert (abs(hy - hx) < 1e-7)
        assert (abs(hz - hx) < 1e-7)
        assert (abs(hz - hy) < 1e-7)
        self.t = T.Texture3D(self.xl, self.yl, self.zl)
        self.s = S.Surface(self.xl,self.yl,self.zl)

class HamiltonianOperatorCuPy:
    Q1 = Q2 = 1.
    """HamiltonianOperatorCuPy. Acts like a linear operator accepting and returning cupy matrices"""

    def __init__(self, x_linspace, y_linspace, z_linspace):
        """__init__.

        :param x_linspace:
        :type x_linspace: np.ndarray
        :param y_linspace:
        :type y_linspace: np.ndarray
        :param z_linspace:
        :type z_linspace: np.ndarray
        """

        self.x_linspace = x_linspace
        self.y_linspace = y_linspace
        self.z_linspace = z_linspace

        self.shape = (len(x_linspace), len(y_linspace), len(z_linspace))

        hx = self.x_linspace[1] - self.x_linspace[0]
        hy = self.y_linspace[1] - self.y_linspace[0]
        hz = self.z_linspace[1] - self.z_linspace[0]
        self.h = hx
        # check if the grid cell is cubic
        assert (abs(hy - hx) < 1e-7)
        assert (abs(hz - hx) < 1e-7)
        assert (abs(hz - hy) < 1e-7)

        self.x_len = len(self.x_linspace)
        self.y_len = len(self.x_linspace)
        self.z_len = len(self.x_linspace)

        N = self.x_len * self.y_len * self.z_len

        self.shape = (N, N)
        self.laplacian = Laplacian3D(h=0.07)
        self.potential = Potential(self.Q1, self.Q2)

        self.t = T.Texture3D(self.x_len, self.y_len, self.z_len)
        self.s = S.Surface(self.x_len, self.y_len, self.z_len)

    def pre(self, v):
        v_cube = v.reshape((self.x_len, self.y_len, self.z_len))

        tex_obj = self.t.texture_from_ndarray(v_cube)

        sur_obj = self.s.initial_surface()

        return tex_obj, sur_obj

    def post(self, sur_out):
        v_out = self.s.get_data(sur_out)
        v = cp.reshape(v_out, (self.x_len * self.y_len * self.z_len))

        return v

    def matvec(self, v: cp.ndarray):
        tex_obj, sur_obj = self.pre(v)
        tex_obj2, sur_obj2 = self.pre(v)
        xl = np.linspace(-1, 1, self.x_len, dtype=np.float32)
        zl = yl = xl
        potential = self.potential.operate_cupy(tex_obj, sur_obj, xl, yl, zl)
        laplacian = self.laplacian.matcube_cupy_27(tex_obj2, sur_obj2)

        v1 = self.post(laplacian)
        v2 = self.post(potential)
        print("kinetic part: ", cp.linalg.norm(v1))
        print("potential part: ", cp.linalg.norm(v2))
        return -0.5 * v1 + v2

    def matmat(self, V: cp.ndarray) -> cp.ndarray:
        V_out = cp.empty_like(V)
        for i in range(cp.size(V, 1)):
            v = V[:, i]
            v_out = self.matvec(v)
            V_out[:, i] = v_out
        return V_out