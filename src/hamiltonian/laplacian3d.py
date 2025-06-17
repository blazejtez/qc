#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Module responsible for the computation of the laplacian of a cube
'''

from typing import Dict, Tuple

import cupy as cp
# import matplotlib.pyplot as plt
import numpy as np
import data_structure.surface as S
import data_structure.texture3D as T
from demo.laplacian_visualisation import plot_laplacian_gto

ALPHA = 0.28294212105225837470023780155114
# ALPHA = 1


_eval_laplacian3d_7pts_stencil_kernel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN,
                                        float h){
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    float value = tex3D<float>(texture_input, idx_x, idx_y, idx_z);

    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        value = -tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z)*6;
        value += (idx_x == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z));
        value += (idx_x == XLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z));
        value += (idx_y == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z));
        value += (idx_y == YLEN-1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z));
        value += (idx_z == 0 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z - 1));
        value += (idx_z == ZLEN -1 ? 0 : tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z + 1));
        value /= h*h;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
                                 }''',
    'test',
    backend='nvcc')

_eval_laplacian3d_19pts_stencil_kernel = cp.RawKernel(
    r'''extern "C" __global__ void laplacian_19pt(cudaTextureObject_t texture_input,
                                                 cudaSurfaceObject_t surface_output, 
                                                 int XLEN, int YLEN, int ZLEN, float h) 
{
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (idx_x >= XLEN || idx_y >= YLEN || idx_z >= ZLEN) return;

    float value = -24 * tex3D<float>(texture_input, idx_x, idx_y, idx_z);

    value += 2 * (idx_x > 0 ? tex3D<float>(texture_input, idx_x - 1, idx_y, idx_z) : 0);
    value += 2 * (idx_x < XLEN - 1 ? tex3D<float>(texture_input, idx_x + 1, idx_y, idx_z) : 0);
    value += 2 * (idx_y > 0 ? tex3D<float>(texture_input, idx_x, idx_y - 1, idx_z) : 0);
    value += 2 * (idx_y < YLEN - 1 ? tex3D<float>(texture_input, idx_x, idx_y + 1, idx_z) : 0);
    value += 2 * (idx_z > 0 ? tex3D<float>(texture_input, idx_x, idx_y, idx_z - 1) : 0);
    value += 2 * (idx_z < ZLEN - 1 ? tex3D<float>(texture_input, idx_x, idx_y, idx_z + 1) : 0);

    value += 1 * ((idx_x > 0 && idx_y > 0) ? tex3D<float>(texture_input, idx_x - 1, idx_y - 1, idx_z) : 0);
    value += 1 * ((idx_x > 0 && idx_y < YLEN - 1) ? tex3D<float>(texture_input, idx_x - 1, idx_y + 1, idx_z) : 0);
    value += 1 * ((idx_x > 0 && idx_z > 0) ? tex3D<float>(texture_input, idx_x - 1, idx_y, idx_z - 1) : 0);
    value += 1 * ((idx_x > 0 && idx_z < ZLEN - 1) ? tex3D<float>(texture_input, idx_x - 1, idx_y, idx_z + 1) : 0);

    value += 1 * ((idx_x < XLEN - 1 && idx_y > 0) ? tex3D<float>(texture_input, idx_x + 1, idx_y - 1, idx_z) : 0);
    value += 1 * ((idx_x < XLEN - 1 && idx_y < YLEN - 1) ? tex3D<float>(texture_input, idx_x + 1, idx_y + 1, idx_z) : 0);
    value += 1 * ((idx_x < XLEN - 1 && idx_z > 0) ? tex3D<float>(texture_input, idx_x + 1, idx_y, idx_z - 1) : 0);
    value += 1 * ((idx_x < XLEN - 1 && idx_z < ZLEN - 1) ? tex3D<float>(texture_input, idx_x + 1, idx_y, idx_z + 1) : 0);

    value += 1 * ((idx_y > 0 && idx_z > 0) ? tex3D<float>(texture_input, idx_x, idx_y - 1, idx_z - 1) : 0);
    value += 1 * ((idx_y > 0 && idx_z < ZLEN - 1) ? tex3D<float>(texture_input, idx_x, idx_y - 1, idx_z + 1) : 0);
    value += 1 * ((idx_y < YLEN - 1 && idx_z > 0) ? tex3D<float>(texture_input, idx_x, idx_y + 1, idx_z - 1) : 0);
    value += 1 * ((idx_y < YLEN - 1 && idx_z < ZLEN - 1) ? tex3D<float>(texture_input, idx_x, idx_y + 1, idx_z + 1) : 0);

    value *= (1.0f / (6.0f * h * h));

    surf3Dwrite<float>(value, surface_output, idx_x * sizeof(float), idx_y, idx_z);
}
    ''',
    'laplacian_19pt',
    backend='nvcc'
)



_eval_laplacian3d_27pts_stencil_kernel_2 = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN,
                                        float h){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx_x >= XLEN || idx_y >= YLEN || idx_z >= ZLEN)
        return;
    float scale = 1.0f / ((30.0f) * h * h) ;
    float center_weight = -128.0f;
    float face_weight = 14.0f;
    float edge_weight = 3.0f;
    float corner_weight = 1.0f;
    float value = tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z) * center_weight;

    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y,     (float)idx_z)     * face_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y,     (float)idx_z)     * face_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x,     (float)idx_y - 1, (float)idx_z)     * face_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x,     (float)idx_y + 1, (float)idx_z)     * face_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x,     (float)idx_y,     (float)idx_z - 1) * face_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x,     (float)idx_y,     (float)idx_z + 1) * face_weight;
        
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z - 1) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z + 1) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y - 1, (float)idx_z) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y + 1, (float)idx_z) * edge_weight;
        
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z - 1) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z - 1) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z + 1) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z + 1) * edge_weight;
        
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z - 1) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z + 1) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y - 1, (float)idx_z) * edge_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y + 1, (float)idx_z) * edge_weight;
        
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y - 1, (float)idx_z - 1) * corner_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y - 1, (float)idx_z + 1) * corner_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y + 1, (float)idx_z - 1) * corner_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y + 1, (float)idx_z + 1) * corner_weight;
        
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y - 1, (float)idx_z - 1) * corner_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y - 1, (float)idx_z + 1) * corner_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y + 1, (float)idx_z - 1) * corner_weight;
        value += (idx_x == 0 || idx_x == XLEN-1 || idx_y == 0 || idx_y == YLEN - 1 || idx_z == 0 || idx_z == ZLEN-1) ? 0 :
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y + 1, (float)idx_z + 1) * corner_weight;
    
        value *= scale;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
                                 }''',
    'test',
    backend='nvcc')


_eval_laplacian3d_27pts_stencil_kernel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                       cudaSurfaceObject_t surface_output,
                                       int XLEN, int YLEN, int ZLEN,
                                       float h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= XLEN || y >= YLEN || z >= ZLEN)
        return;
    float scale = 1.0f / ((30.0f) * h * h) ;
    float center_weight = -128.0f;
    float face_weight = 14.0f;
    float edge_weight = 3.0f;
    float corner_weight = 1.0f;
    float laplacian = center_weight * tex3D<float>(texture_input, x, y, z);
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                float value = tex3D<float>(texture_input, x + dx, y + dy, z + dz);
                float weight;
                int neighbors = abs(dx) + abs(dy) + abs(dz);
                if (neighbors == 1) {
                    weight = face_weight;
                } else if (neighbors == 2) {
                    weight = edge_weight;
                } else {
                    weight = corner_weight;
                }
                laplacian += weight * value;
            }
        }
    }
    laplacian *= scale;
    surf3Dwrite(laplacian, surface_output, x * sizeof(float), y, z);
}''',
    'test',
    backend='nvcc')

class Laplacian3D:
    BLOCKSIZE = 8

    def __init__(self, h: np.float32 = 1.):
        self.h = h
        _eval_laplacian3d_7pts_stencil_kernel.compile()
        _eval_laplacian3d_27pts_stencil_kernel_2.compile()
        _eval_laplacian3d_19pts_stencil_kernel.compile()

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

    def matcube_cupy_7(self, texture, surface):
        xlen = texture.ResDesc.cuArr.width
        ylen = texture.ResDesc.cuArr.height
        zlen = texture.ResDesc.cuArr.depth
        _eval_laplacian3d_7pts_stencil_kernel(
            (np.int32(np.ceil(float(xlen) / Laplacian3D.BLOCKSIZE)),
             np.int32(np.ceil(float(ylen) / Laplacian3D.BLOCKSIZE)),
             np.int32(np.ceil(float(zlen) / Laplacian3D.BLOCKSIZE))),
            (Laplacian3D.BLOCKSIZE, Laplacian3D.BLOCKSIZE, Laplacian3D.BLOCKSIZE), (
                texture,
                surface,
                xlen,
                xlen,
                zlen,
                np.float32(self.h)
            ))
        return surface

    def matcube_cupy_19(self, texture, surface):
        xlen = texture.ResDesc.cuArr.width
        ylen = texture.ResDesc.cuArr.height
        zlen = texture.ResDesc.cuArr.depth
        _eval_laplacian3d_19pts_stencil_kernel(
            (np.int32(np.ceil(float(xlen) / Laplacian3D.BLOCKSIZE)),
             np.int32(np.ceil(float(ylen) / Laplacian3D.BLOCKSIZE)),
             np.int32(np.ceil(float(zlen) / Laplacian3D.BLOCKSIZE))),
            (Laplacian3D.BLOCKSIZE, Laplacian3D.BLOCKSIZE, Laplacian3D.BLOCKSIZE), (
                texture,
                surface,
                xlen,
                xlen,
                zlen,
                np.float32(self.h)
            ))
        return surface

    def matcube_cupy_27(self, texture, surface) -> cp.ndarray:
        xlen = texture.ResDesc.cuArr.width
        ylen = texture.ResDesc.cuArr.height
        zlen = texture.ResDesc.cuArr.depth
        _eval_laplacian3d_27pts_stencil_kernel_2(
            (np.int32(np.ceil(float(xlen) / Laplacian3D.BLOCKSIZE)),
             np.int32(np.ceil(float(ylen) / Laplacian3D.BLOCKSIZE)),
             np.int32(np.ceil(float(zlen) / Laplacian3D.BLOCKSIZE))),
            (Laplacian3D.BLOCKSIZE, Laplacian3D.BLOCKSIZE, Laplacian3D.BLOCKSIZE), (
                texture,
                surface,
                xlen,
                xlen,
                zlen,
                np.float32(self.h)
            ))
        return surface

    # def matcube_analytical(self, texture, surface) -> cp.ndarray:
    #     xlen = texture.ResDesc.cuArr.width
    #     ylen = texture.ResDesc.cuArr.height
    #     zlen = texture.ResDesc.cuArr.depth
    #     y = np.linspace(-10, 10, ylen)
    #     x = np.linspace(-10, 10, xlen)
    #     z = np.linspace(-10, 10, zlen)
    #     X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    #     a = laplacian_gto(ALPHA, X, Y, Z)
    #     cp_a = cp.array(a)
    #     return cp_a.reshape((xlen * ylen * zlen, 1))

    def _lower_vec(self, k: np.ndarray):
        return np.maximum(k, np.array([0, 0, 0]))

    def _upper_vec(self, k: np.ndarray):
        return np.minimum(self.shape + k, self.shape)

    def _lower_vec_out(self, k: np.ndarray):
        return np.maximum(-k, np.array([0, 0, 0]))

    def _upper_vec_out(self, k: np.ndarray):
        return np.minimum(self.shape - k, self.shape)

#
# if __name__ == "__main__":
#
#     # st = Stencils3D()
#     #
#     # shape = (1000, 1000, 1000)
#     #
#     lap = Laplacian3D(h=0.1)
#     #
#     # cube = np.random.randn(*shape).astype(np.float32)
#     # tic = time.time()
#     # cube_out = lap.matcube(cube, st.stencil7pts)
#     # toc = time.time()
#     # print(f"time elapsed numpy: {toc-tic}")
#     #
#     # tic = time.time()
#     # cube_out_numba = lap.matcube_numba(cube)
#     # toc = time.time()
#     # print(f"time elapsed numba: {toc-tic}")
#     #
#     # tic = time.time()
#     # cube_out_numba = lap.matcube_numba(cube)
#     # toc = time.time()
#     # print(f"time elapsed numba: {toc-tic}")
#     # print((cube_out-cube_out_numba)**2)
#     start_event = cp.cuda.stream.Event()
#     stop_event = cp.cuda.stream.Event()
#     stream = cp.cuda.stream.Stream()
#     tex_obj = T.Texture3D(10, 10, 10)
#     sur_obj = S.Surface(10, 10, 10)
#     # init_tex = tex_obj.initial_texture()
#     init_tex = tex_obj.ones()
#     init_sur = sur_obj.initial_surface()
#     print(cp.gradient(cp.gradient(cp.ones((10, 10, 10), dtype=cp.float32))))
#     with stream:
#         start_event.record()
#         sur = lap.matcube_cupy_27(init_tex, init_sur)
#         stop_event.record()
#         stop_event.synchronize()
#         print(f"time elapsed cuda27: {cp.cuda.stream.get_elapsed_time(start_event, stop_event) * 1e-3}")
#     for i in sur_obj.get_data(init_sur):
#         print(i)
#     # plt.plot(sur_obj.get_data(init_sur))
#     # plt.show()


if __name__ == "__main__":
    # Step 3: Create input data
    XLEN, YLEN, ZLEN = 601, 601, 601  # Dimensions of the input arrays and grid
    eps = 1e-10  # Small epsilon to prevent division by zero
    extent = 10.0  # Extent of the grid in Bohr radii

    # Coordinates (centered around the nucleus at (0, 0, 0))

    xl = np.linspace(-extent, extent, XLEN, dtype=cp.float32)
    yl = np.linspace(-extent, extent, YLEN, dtype=cp.float32)
    zl = np.linspace(-extent, extent, ZLEN, dtype=cp.float32)

    hx = xl[1] - xl[0]
    print(hx)
    X, Y, Z = cp.meshgrid(cp.array(xl), cp.array(yl), cp.array(zl))
    gaussian_orbital = cp.exp(-1*(X ** 2 + Y ** 2 + Z ** 2), dtype=cp.float32)

    tex_obj = T.Texture3D(len(xl),len(yl),len(zl))
    sur_obj = S.Surface(len(xl),len(yl),len(zl))
    init_tex = tex_obj.texture_from_ndarray(gaussian_orbital)
    init_sur = sur_obj.initial_surface()
    # Step 4: Launch the kernel
    blocksize = (8, 8, 8)  # Assuming BLOCKSIZE in the kernel is 8
    gridsize = (int((XLEN + blocksize[0] - 1) / blocksize[0]),
                int((YLEN + blocksize[1] - 1) / blocksize[1]),
                int((ZLEN + blocksize[2] - 1) / blocksize[2]))
    l = Laplacian3D(h=hx)
    l.matcube_cupy_27(init_tex, init_sur)

    # Step 5: Check the results
    result = sur_obj.get_data(init_sur)  # Move the result back to host (CPU) for inspection
    result_cpu = cp.asnumpy(result)

    # # Calculate analytical solution of laplacian for comparison
    # Hv = plot_laplacian_gto(ALPHA, XLEN, extent)
    # Hv = Hv.reshape((XLEN*YLEN*ZLEN,1))
    # v = gaussian_orbital.flatten()
    # vtv = v.T.dot(v)
    # vtHv = v.T.dot(Hv)
    # print(vtHv/vtv)
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(np.sum(np.sum(result_cpu, axis=1), axis=1))  # Plot the data
    plt.xlabel('Grid steps, 1 bohr radius=30 grid steps')
    plt.ylabel('Laplacian value a.u.')
    plt.savefig('laplacian1d.png', bbox_inches='tight')

    # plt.figure()
    # plt.plot(result_cpu[250][300])
    # plt.title(f'Laplacian 1D slice')
    # plt.xlabel('Grid steps, 1 bohr radius=30 grid steps')
    # plt.ylabel('Laplacian value a.u.')
    # plt.show()

    #
    # plt.figure()
    # plt.plot(result_cpu.sum(axis=0).sum(axis=0) - Hv.sum(axis=0).sum(axis=0).get())
    # plt.title(f'Difference between FDM laplacian and analytical laplacian')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.show()

    # # Print the result for inspection
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