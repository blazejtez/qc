#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cupy as cp
from matplotlib import pyplot as plt

import src.qc.texture as T
import src.qc.surface as S
import time
from numba import njit, jit, prange

_eval_potential_kernel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, float *xl, float *yl, float *zl, 
                                        int XLEN, int YLEN, int ZLEN, double Z1, double Z2, double eps) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
        
     if((idx_x >= XLEN) || (idx_y >= YLEN) || (idx_z >= ZLEN)){
        return;  // Ensure that the threads are within bounds
    }
    
    float value = 0;
    float aux = 0;
    float aux2 = 0.0;
    
    #define BLOCKSIZE 8 //have to be same like the Potential.BLOCKSIZE class attribute
    __shared__ float xls[BLOCKSIZE];
    __shared__ float yls[BLOCKSIZE];
    __shared__ float zls[BLOCKSIZE];
    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        xls[threadIdx.x] = xl[idx_x];
        yls[threadIdx.y] = yl[idx_y];
        zls[threadIdx.z] = zl[idx_z];
        value = tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z);
        aux = xls[threadIdx.x]*xls[threadIdx.x] + yls[threadIdx.y]*yls[threadIdx.y] + zls[threadIdx.z]*zls[threadIdx.z];
        aux2 = Z1 * Z2 *__frsqrt_rn(aux < eps ? eps : aux);
        value *= aux2;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
                                 }''',
    'test',
    backend='nvcc')

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
             z_linspace: np.ndarray, Z1: float, Z2: float, eps: float = 1e-4):

    xlen = len(x_linspace)
    ylen = len(y_linspace)
    zlen = len(z_linspace)

    cube_out = np.empty_like(cube)

    for x in prange(xlen):
        xval = x_linspace[x]
        x2 = xval**2
        for y in prange(ylen):
            yval = y_linspace[y]
            x2y2 = x2 + yval**2
            for z in prange(zlen):
                zval = z_linspace[z]
                x2y2z2 = x2y2+zval**2
                val = cube[x, y, z]  * Z1 * Z2 * 1 / np.sqrt(max(x2y2z2,eps))
                cube_out[x,y,z] = val

    return cube_out
class Potential:
    BLOCKSIZE = 8    
    eps = 1e-4
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

        _eval_potential_kernel.compile()

    def evaluate(self, xl, yl, zl):

        return _evaluate(xl, yl, zl, self.Q1, self.Q2)

    def operate(self, cube, xl, yl, zl):
        assert(np.size(cube,0) == len(xl))
        assert(np.size(cube,1) == len(yl))
        assert(np.size(cube,2) == len(zl))
        cube_out = _operate(cube, xl,yl, zl, self.Q1, self.Q2, Potential.eps)

        return cube_out 
    def operate_cupy(self, texture, surface, xl, yl, zl):
        
        xlen = texture.ResDesc.cuArr.width 
        ylen = texture.ResDesc.cuArr.height
        zlen = texture.ResDesc.cuArr.depth
        xl = cp.asarray(xl)
        yl = cp.asarray(yl)
        zl = cp.asarray(zl)
        _eval_potential_kernel(
            (np.int32(np.ceil(float(xlen)/Potential.BLOCKSIZE)),
             np.int32(np.ceil(float(ylen)/Potential.BLOCKSIZE)), 
             np.int32(np.ceil(float(zlen)/Potential.BLOCKSIZE)) ), 
            (Potential.BLOCKSIZE,Potential.BLOCKSIZE,Potential.BLOCKSIZE ), (
            texture,
            surface,
            xl,
            yl,
            zl,
            xlen,
            xlen,
            zlen,
            np.float64(self.Q1),
            np.float64(self.Q2),
            np.float64(Potential.eps)
        ))
        return surface
#
# if __name__ == "__main__":
#
#     xl = np.linspace(-1, 1, 1001, dtype=np.float32)
#     yl = np.linspace(-1, 1, 1001,  dtype=np.float32)
#     zl = np.linspace(-1, 1, 1001, dtype=np.float32)
#     print('start generating random cube ...')
#     tic = time.time()
#     cube = np.random.randn(len(xl),len(yl),len(zl)).astype(np.float32)
#     toc = time.time()
#     print(f"time elapsed: {toc-tic} sec.")
#     p = Potential()
#     print('start computation of potential ...')
#     tic = time.time()
#     out = p.operate(cube, xl, yl, zl)
#     toc = time.time()
#     print(f'time elapsed numba: {toc-tic} sec.')
#
#     print('start computation of potential ...')
#     tic = time.time()
#     out = p.operate(cube, xl, yl, zl)
#     toc = time.time()
#     print(f'time elapsed numba: {toc-tic} sec.')
#
#     start_event = cp.cuda.stream.Event()
#     stop_event = cp.cuda.stream.Event()
#     stream = cp.cuda.stream.Stream()
#     tex_obj = T.Texture(len(xl),len(yl),len(zl))
#     sur_obj = S.Surface(len(xl),len(yl),len(zl))
#     init_tex = tex_obj.initial_texture()
#     init_sur = sur_obj.initial_surface()
#     with stream:
#         start_event.record()
#         sur = p.operate_cupy(init_tex, init_sur, xl, yl, zl)
#         stop_event.record()
#         stop_event.synchronize()
#         print(f"time elapsed cuda: {cp.cuda.stream.get_elapsed_time(start_event,stop_event)*1e-3}")

# Step 3: Create input data
XLEN, YLEN, ZLEN = 64, 64, 64  # Dimensions of the input arrays and grid
Z1, Z2 = 1.0, -1.0  # Charges in atomic units
eps = 1e-10  # Small epsilon to prevent division by zero
extent = 10.0  # Extent of the grid in Bohr radii (5 Bohr radii in each direction)

# Coordinates (centered around the nucleus at (0, 0, 0))
xl = cp.linspace(-extent/2, extent/2, XLEN, dtype=cp.float32)
yl = cp.linspace(-extent/2, extent/2, YLEN, dtype=cp.float32)
zl = cp.linspace(-extent/2, extent/2, ZLEN, dtype=cp.float32)


tex_obj = T.Texture(len(xl),len(yl),len(zl))
sur_obj = S.Surface(len(xl),len(yl),len(zl))
init_tex = tex_obj.initial_texture()
init_sur = sur_obj.initial_surface()
# Step 4: Launch the kernel
blocksize = (8, 8, 8)  # Assuming BLOCKSIZE in the kernel is 8
gridsize = (int((XLEN + blocksize[0] - 1) / blocksize[0]),
            int((YLEN + blocksize[1] - 1) / blocksize[1]),
            int((ZLEN + blocksize[2] - 1) / blocksize[2]))

for i in range(50):
    _eval_potential_kernel(gridsize, blocksize, (init_tex, init_sur, xl, yl, zl, XLEN, YLEN, ZLEN, Z1, Z2, eps))
    init_tex = tex_obj.texture_from_surface(init_sur)

# Step 5: Check the results
result = sur_obj.get_data(init_sur)  # Move the result back to host (CPU) for inspection
result_cpu = cp.asnumpy(result)
# Print the result for inspection
print("Output Data:")
print(result_cpu[3])
for i in range(YLEN):
    plt.figure(figsize=(64, 64))
    plt.imshow(result_cpu[i], cmap='viridis', origin='lower')
    plt.colorbar(label='Output value')
    plt.title(f'Output data result at Z=4')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()
    time.sleep(0.01)