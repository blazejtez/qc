#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Module responsible for the computation of the laplacian of a cube
'''

import time
from typing import Dict, Tuple

import cupy as cp
# import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from numba import jit, prange
import src.qc.surface as S
import src.qc.texture as T

_eval_laplacian3d_7pts_stencil_kernel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN,
                                        float h){
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    float value = 0;
    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < uLEN)){
        value = -tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z)*6 + \
        (idx_x == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z)) + \
        (idx_x == XLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z)) + \
        (idx_y == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z)) + \
        (idx_y == YLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z)) + \
        (idx_z == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z - 1)) + \
        (idx_z == ZLEN -1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z + 1));
        value *= h;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
                                 }''',
    'test',
    backend='nvcc')

_eval_laplacian3d_27pts_stencil_kernel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN,
                                        float h){
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    float value = 0;
    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        value = -tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z)*6 + \
        (idx_x == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z)) + \
        (idx_x == XLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z)) + \
        (idx_y == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z)) + \
        (idx_y == YLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z)) + \
        (idx_z == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z - 1)) + \
        (idx_z == ZLEN -1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z + 1));
        value *= h;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
                                 }''',
    'test',
    backend='nvcc')

@jit(nopython=False, parallel=True)
def _eval_laplacian3d_7pts_stencil(cube: np.ndarray, xlen: int, ylen: int,
                                   zlen: int, h: float) -> np.ndarray:
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

    cube_out[xlen - 1, ylen - 1,
             zlen - 1] = -6 * cube[xlen - 1, ylen - 1, zlen - 1]
    cube_out[xlen - 1, ylen - 1, zlen -
             1] += cube[xlen - 2, ylen - 1, zlen -
                        1] + cube[xlen - 1, ylen - 2, zlen -
                                  1] + cube[xlen - 1, ylen - 1, zlen - 2]

    # multiply times the cube of the delta between neighboring grid points
    for x in prange(xlen):
        for y in prange(ylen):
            for z in prange(zlen):
                cube_out[x, y, z] *= h

    return cube_out


class Stencils3D:
    """Stencils3D."""
    def __init__(self):

        self.stencil7pts = {
            (0, 0, 0): -6,
            (-1, 0, 0): 1,
            (1, 0, 0): 1,
            (0, -1, 0): 1,
            (0, 1, 0): 1,
            (0, 0, -1): 1,
            (0, 0, 1): 1
        }


class Laplacian3D:
    BLOCKSIZE = 8
    def __init__(self, h: float = 1.):
        self.h = h

        _eval_laplacian3d_7pts_stencil_kernel.compile()


    def matcube(self, cube: np.ndarray, stencil: Dict[Tuple[int, int, int],
                                                      float]) -> np.ndarray:
        xlen = np.size(cube, 0)
        ylen = np.size(cube, 1)
        zlen = np.size(cube, 2)
        self.shape = (xlen, ylen, zlen)

        cube_out = np.zeros(self.shape, dtype=np.float32)
        for k, v in stencil.items():
            k_np = np.asarray(k)
            lv = self._lower_vec(k_np)
            uv = self._upper_vec(k_np)
            lvo = self._lower_vec_out(k_np)
            uvo = self._upper_vec_out(k_np)
            cube_out[lvo[0]:uvo[0], lvo[1]:uvo[1],
                     lvo[2]:uvo[2]] += v * cube[lv[0]:uv[0], lv[1]:uv[1],
                                                lv[2]:uv[2]] * self.h

        return cube_out

    def matcube_numba(self, cube: np.ndarray) -> np.ndarray:
        xlen = np.size(cube, 0)
        ylen = np.size(cube, 1)
        zlen = np.size(cube, 2)
        return _eval_laplacian3d_7pts_stencil(cube, xlen, ylen, zlen, self.h)

    def matcube_cupy(self, texture, surface):
        xlen = texture.ResDesc.cuArr.width 
        ylen = texture.ResDesc.cuArr.height
        zlen = texture.ResDesc.cuArr.depth
        _eval_laplacian3d_7pts_stencil_kernel(
            (np.int32(np.ceil(float(xlen)/Laplacian3D.BLOCKSIZE)),
             np.int32(np.ceil(float(ylen)/Laplacian3D.BLOCKSIZE)), 
             np.int32(np.ceil(float(zlen)/Laplacian3D.BLOCKSIZE)) ), 
            (Laplacian3D.BLOCKSIZE,Laplacian3D.BLOCKSIZE,Laplacian3D.BLOCKSIZE ), (
            texture,
            surface,
            xlen,
            xlen,
            zlen,
            np.float32(self.h)
        ))
        return surface

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

    shape = (1000, 1000, 1000)

    lap = Laplacian3D(h=0.1)

    cube = np.random.randn(*shape).astype(np.float32)
    tic = time.time()
    cube_out = lap.matcube(cube, st.stencil7pts)
    toc = time.time()
    print(f"time elapsed numpy: {toc-tic}")

    tic = time.time()
    cube_out_numba = lap.matcube_numba(cube)
    toc = time.time()
    print(f"time elapsed numba: {toc-tic}")

    tic = time.time()
    cube_out_numba = lap.matcube_numba(cube)
    toc = time.time()
    print(f"time elapsed numba: {toc-tic}")
    #print((cube_out-cube_out_numba)**2)
    start_event = cp.cuda.stream.Event()
    stop_event = cp.cuda.stream.Event()
    stream = cp.cuda.stream.Stream()
    tex_obj = T.Texture(1000,1000,1000)
    sur_obj = S.Surface(1000,1000,1000)
    init_tex = tex_obj.initial_texture()
    init_sur = sur_obj.initial_surface()
    with stream:
        start_event.record()
        sur = lap.matcube_cupy(init_tex, init_sur)
        stop_event.record()
        stop_event.synchronize()
        print(f"time elapsed cuda: {cp.cuda.stream.get_elapsed_time(start_event,stop_event)*1e-3}")
    # plt.plot(cube_out.flatten())
    # plt.plot(cube_out_numba.flatten())
    # plt.show()
