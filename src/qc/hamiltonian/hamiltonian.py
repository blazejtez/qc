#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    def __init__(self, x_linspace, y_linspace, z_linspace, extent):
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
        self.extent = extent

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
        self.laplacian = Laplacian3D(h=self.h)
        self.potential = Potential(self.x_linspace, self.Q1, self.Q2, extent=self.extent)

        self.t = T.Texture3D(self.x_len, self.y_len, self.z_len)
        self.s = S.Surface(self.x_len, self.y_len, self.z_len)

    def pre(self, v):
        v_cube = v.reshape((self.x_len, self.y_len, self.z_len))

        tex_obj = self.t.texture_from_ndarray(v_cube)

        sur_obj = self.s.initial_surface()

        return tex_obj, sur_obj

    def post(self, sur_out):
        v_out = self.s.get_data(sur_out)
        v = cp.reshape(v_out, (self.x_len * self.y_len * self.z_len, 1),)

        return v

    def matvec(self, v: cp.ndarray):
        tex_obj, sur_obj = self.pre(v)
        # v1 = self.potential.truncated_potential_3d()
        potential = self.potential.operate_cupy(tex_obj, sur_obj, self.x_linspace, self.y_linspace, self.z_linspace)
        v1 = self.post(potential)

        tex_obj, sur_obj = self.pre(v)
        laplacian = self.laplacian.matcube_cupy_27(tex_obj, sur_obj)
        v2 = self.post(laplacian)
        # print("kinetic part: ", cp.sum(v1))
        # print("potential part: ", cp.sum(v2))
        return -v1 - 0.5*v2

    def matmat(self, V: cp.ndarray) -> cp.ndarray:
        V_out = cp.empty_like(V)
        for i in range(cp.size(V, 1)):
            v = V[:, i]
            v_out = self.matvec(v)
            V_out[:, i] = v_out
        return V_out

if __name__ == "__main__":
    ALPHA = 0.28294212105225837470023780155114
    # Step 3: Create input data
    XLEN, YLEN, ZLEN = 600, 600, 600  # Dimensions of the input arrays and grid
    extent = 10

    # Coordinates (centered around the nucleus at (0, 0, 0))
    xl = np.linspace(-extent, extent, XLEN, dtype=cp.float32)
    yl = np.linspace(-extent, extent, YLEN, dtype=cp.float32)
    zl = np.linspace(-extent, extent, ZLEN, dtype=cp.float32)
    h = HamiltonianOperatorCuPy(xl, yl, zl, extent=extent)
    dx = xl[1] - xl[0]
    X, Y, Z = cp.meshgrid(cp.array(xl), cp.array(yl), cp.array(zl))
    gaussian_orbital = cp.exp(-ALPHA * (X ** 2 + Y ** 2 + Z ** 2), dtype=cp.float32)
    norm_factor = cp.sum(gaussian_orbital ** 2) * h.h ** 3
    gaussian_orbital /= cp.sqrt(norm_factor)


    v = cp.reshape(gaussian_orbital, (len(xl) * len(yl) * len(zl), 1))
    # # v = cp.random.randn((len(xl) * len(yl) * len(zl)), 1, dtype=cp.float32)
    # # Step 4: Launch the kernel
    print(v.T.dot(h.matvec(v)))
    print((v.T.dot(v)))
    result = v.T.dot(h.matvec(v))/(v.T.dot(v))
    print("real result", result)
    # result = result.reshape((len(xl), len(yl), len(zl)))
    # result_cpu = cp.asnumpy(result)

    # Print the result for inspection
    # print(result_cpu)

    # plt.figure(figsize=(64, 64))
    # plt.imshow(result_cpu.sum(axis=0), cmap='viridis', origin='lower')
    # plt.colorbar(label='Output value')
    # plt.title(f'Output data result 2D summed on x axis')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.show()
    #
    # plt.figure(figsize=(64, 64))
    # plt.imshow(result_cpu.sum(axis=1), cmap='viridis', origin='lower')
    # plt.colorbar(label='Output value')
    # plt.title(f'Output data result 2D summed on x axis')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.show()
    #
    # plt.figure(figsize=(64, 64))
    # plt.imshow(result_cpu.sum(axis=2), cmap='viridis', origin='lower')
    # plt.colorbar(label='Output value')
    # plt.title(f'Output data result 2D summed on x axis')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.show()
    #
    #
    # plt.figure()
    # plt.plot(result_cpu.sum(axis=0).sum(axis=0))
    # plt.title(f'Output 1D summed')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.show()

    total = -cp.sum(result) * (h.h ** 3)

    print("total energy:", total)

