#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time

import cupy as cp
import numpy as np
from numba import jit, prange

import qc.surface as S
import qc.texture as T
from qc.laplacian3d import *
from qc.potential import *

_eval_hamiltonian_unperturbed_kernel = cp.RawKernel(
    r'''extern "C" __global__ void hamiltonian(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, float *xl, float *yl, float *zl, 
                                        int XLEN, int YLEN, int ZLEN, double Z1, double Z2, double eps, double h) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    float value = 0;
    float aux = 0;
    float aux1 = 0;
    float aux2 = 0;
    //#define BLOCKSIZE 8 //have be same like the HydrogenHamiltonian.BLOCKSIZE class atribute
    //__shared__ float xls[BLOCKSIZE];
    //__shared__ float yls[BLOCKSIZE];
    //__shared__ float zls[BLOCKSIZE];

    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        //xls[threadIdx.x] = xl[idx_x];
        //yls[threadIdx.y] = yl[idx_y];
        //zls[threadIdx.z] = zl[idx_z];
        value = tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z);
        aux = -6*value +\
        (idx_x == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z)) + \
        (idx_x == XLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z)) + \
        (idx_y == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z)) + \
        (idx_y == YLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z)) + \
        (idx_z == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z - 1)) + \
        (idx_z == ZLEN -1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z + 1));
        aux *= h;
        aux1 = xl[idx_x]*xl[idx_x] + yl[idx_y]*yl[idx_y] + zl[idx_z]*zl[idx_z];
        aux2 = Z1 * Z2 *__frsqrt_rn(aux1 < eps ? eps : aux1);
        value = -value * aux2 - 0.5 * aux;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
    }''',
    'hamiltonian',
    backend='nvcc')


@jit(nopython=True, parallel=True)
def _add_cubes_hamiltonian(c1laplacian: np.ndarray, c2potential: np.ndarray,
                           xlen: int, ylen: int, zlen: int):
    out = np.empty_like(c1laplacian)
    for x in prange(xlen):
        for y in prange(ylen):
            for z in prange(zlen):
                out[x, y, z] = c2potential[x, y, z] - .5 * c1laplacian[x, y, z]

    return out


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

        self.p = Potential()
        self.xl = x_linspace
        self.yl = y_linspace
        self.zl = z_linspace

        self.shape = (len(x_linspace), len(y_linspace), len(z_linspace))

        hx = self.xl[1] - self.xl[0]
        hy = self.yl[1] - self.yl[0]
        hz = self.zl[1] - self.zl[0]
        self.h = hx**(-1)
        # check if the grid cell is cubic
        assert (abs(hy - hx) < 1e-7)
        assert (abs(hz - hx) < 1e-7)
        assert (abs(hz - hy) < 1e-7)

        self.l = Laplacian3D(h=self.h)

        _eval_hamiltonian_unperturbed_kernel.compile()

    def operate(self, cube: np.ndarray) -> np.ndarray:

        cube_l = self.l.matcube_numba(cube)

        cube_p = self.p.operate(cube, self.xl, self.yl, self.zl)

        xlen = np.size(cube, 0)
        ylen = np.size(cube, 1)
        zlen = np.size(cube, 2)

        return _add_cubes_hamiltonian(cube_l, cube_p, xlen, ylen, zlen)

    def operate_vec(self, vecs: np.ndarray) -> np.ndarray:

        vecs_copy = copy.copy(vecs)

        for i in range(np.size(vecs, 1)):

            cube = np.reshape(vecs_copy[:, i], self.shape)
            cube = self.operate(cube)
            vecs_copy[:, i] = np.reshape(cube, (np.prod(self.shape), ))

        return vecs_copy

    def operate_unperturbed_cupy(self, texture, surface, Q1, Q2):

        xlen = texture.ResDesc.cuArr.width
        ylen = texture.ResDesc.cuArr.height
        zlen = texture.ResDesc.cuArr.depth
        xl = cp.asarray(self.xl, dtype=cp.float32)
        yl = cp.asarray(self.yl, dtype=cp.float32)
        zl = cp.asarray(self.zl, dtype=cp.float32)
        _eval_hamiltonian_unperturbed_kernel(
            (np.int32(np.ceil(float(xlen) / HydrogenHamiltonian.BLOCKSIZE)),
             np.int32(np.ceil(float(ylen) / HydrogenHamiltonian.BLOCKSIZE)),
             np.int32(np.ceil(float(zlen) / HydrogenHamiltonian.BLOCKSIZE))),
            (HydrogenHamiltonian.BLOCKSIZE, HydrogenHamiltonian.BLOCKSIZE,
             HydrogenHamiltonian.BLOCKSIZE),
            (texture, surface, xl, yl, zl, xlen, xlen, zlen, np.float64(Q1), np.float64(Q2), HydrogenHamiltonian.eps, np.float64(
                self.h)))
        return surface

class HamiltonianOperatorNumPy:
    Q1 = Q2 = 1.
    """HamiltonianOperatorNumPy. Acts like a linear operator accepting and returning numpy matrices"""


    def __init__(self, xl, yl, zl):
        """__init__.

        :param xl: linspace for x axis
        :param yl: linspace for y axis
        :param zl: linspace for z axis
        """

        self.h = HydrogenHamiltonian(xl,yl,zl)

        
        self.x_len = len(self.h.xl)
        self.y_len = len(self.h.yl)
        self.z_len = len(self.h.zl)

        N = self.x_len*self.y_len*self.z_len

        self.shape = (N,N)

        self.t = T.Texture(self.x_len,self.y_len,self.z_len)
        self.s = S.Surface(self.x_len, self.y_len, self.z_len)

    def pre(self, v):

        v_cube = v.reshape((self.x_len,self.y_len,self.z_len) )
        
        tex_obj = self.t.texture_from_ndarray(v_cube)

        sur_obj = self.s.initial_surface()

        return tex_obj, sur_obj

    def post(self, sur_out):

        v_out = self.s.get_data(sur_out)

        v_out_numpy = cp.asnumpy(v_out)

        v = np.reshape(v_out_numpy, (self.x_len * self.y_len * self.z_len,1))

        return v
        
    def matvec(self, v):
        
        tex_obj, sur_obj = self.pre(v)

        sur_out = self.h.operate_unperturbed_cupy(tex_obj,sur_obj,HamiltonianOperatorNumPy.Q1, HamiltonianOperatorNumPy.Q2)        

        v = self.post(sur_out)

        return v
    
    def matmat(self, V : np.ndarray) -> np.ndarray:

        V_out = np.empty_like(V)

        for i in range(np.size(V,1)):

            v = V[:,i]

            v_out = self.matvec(v)

            V_out[:,i] = v_out

        return V_out

class HamiltonianSquaredOperatorNumPy(HamiltonianOperatorNumPy):

    def __init__(self, xl, yl, zl):

        super().__init__(xl,yl,zl)

    def matvec(self, v : np.ndarray) -> np.ndarray:

        print("Matvec in HamiltonianSquaredOperatorNumPy ...")

        tex_obj, sur_obj = self.pre(v) 

        sur_out = self.h.operate_unperturbed_cupy(tex_obj,sur_obj,HamiltonianOperatorNumPy.Q1, HamiltonianOperatorNumPy.Q1)     
        
        tex_obj = self.t.texture_from_surface(sur_out)

        sur_out = self.h.operate_unperturbed_cupy(tex_obj, sur_obj, HamiltonianOperatorNumPy.Q1, HamiltonianOperatorNumPy.Q2)

        v = self.post(sur_out)

        return v

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

    #  vecs = np.random.randn(np.prod((ln, ln, ln)), 3).astype(np.float32)
    #  print("start applying Hamiltonian ...")
    #  tic = time.time()
    #  vecs = h.operate_vec(vecs)
    #  toc = time.time()
    #  print(f"time elapsed: {toc-tic} sec.")
    #
    #  vecs = np.random.randn(np.prod((ln, ln, ln)), 3).astype(np.float32)
    #  print("start applying Hamiltonian ...")
    #  tic = time.time()
    #  vecs = h.operate_vec(vecs)
    #  toc = time.time()
    #  print(f"time elapsed: {toc-tic} sec.")


    start_event = cp.cuda.stream.Event()
    stop_event = cp.cuda.stream.Event()
    stream = cp.cuda.stream.Stream()
    tex_obj = T.Texture(ln,ln,ln)
    sur_obj = S.Surface(ln,ln,ln)
    init_tex = tex_obj.initial_texture()
    init_sur = sur_obj.initial_surface()
    with stream:
        start_event.record()
        sur = h.operate_unperturbed_cupy(init_tex, init_sur, 1., 1.)
        stop_event.record()
        stop_event.synchronize()
        print(f"time elapsed cuda: {cp.cuda.stream.get_elapsed_time(start_event,stop_event)*1e-3}")
