#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cupy as cp
from matplotlib import pyplot as plt

import data_structure.surface as S
import data_structure.texture3D as T

_eval_potential_kernel = cp.RawKernel(
    r'''extern "C" __global__ void potential(cudaTextureObject_t texture_input,
                                            cudaSurfaceObject_t surface_output,
                                            float *xl, float *yl, float *zl,
                                            int XLEN, int YLEN, int ZLEN,
                                            double Z1, double Z2, double eps) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    float value = 0;
    float potential = 0;
    #define BLOCKSIZE 8 //have be same like the Potential.BLOCKSIZE class atribute
    __shared__ float xls[BLOCKSIZE];
    __shared__ float yls[BLOCKSIZE];
    __shared__ float zls[BLOCKSIZE];
    if((idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){
        xls[threadIdx.x] = xl[idx_x];
        yls[threadIdx.y] = yl[idx_y];
        zls[threadIdx.z] = zl[idx_z];
        value = tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z);
        float r_squared = xls[threadIdx.x]*xls[threadIdx.x] + yls[threadIdx.y]*yls[threadIdx.y] + zls[threadIdx.z]*zls[threadIdx.z];
        float r = __frsqrt_rn(r_squared < eps ? eps : r_squared);
        value *= r;
        surf3Dwrite<float>(value, surface_output,idx_x*sizeof(float),idx_y,idx_z);
    }
    }''',
    'potential',
    backend='nvcc'
)


class Potential:
    BLOCKSIZE = 8    
    eps = 1e-2
    """Potential. Computes Coulomb potential for a 3D mesh"""
    def __init__(self, x_linspace: np.ndarray, Q1: float = 1, Q2: float = 1, extent: float = 10):
        """__init__.

        :param Q1: charge of the first particle in atomic units
        :type Q1: float
        :param Q2: charge of the second particle in atomic units
        :type Q2: float
        """
        self.Q1 = Q1
        self.Q2 = Q2
        self.extent = extent
        self.N = len(x_linspace)
        # TODO: proper calculation of v0
        # self.v0 = self.N / extent
        self.v0 = 1

        _eval_potential_kernel.compile()

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
        surface_factory = S.Surface(xlen, ylen, zlen)
        return surface


    def truncated_potential_3d(self):
        """
        Calculate the truncated potential in 3D for a given grid.

        Parameters:
        x (cupy.ndarray): 1D array representing the x-coordinates.
        y (cupy.ndarray): 1D array representing the y-coordinates.
        z (cupy.ndarray): 1D array representing the z-coordinates.
        V0 (float): The maximum potential value.

        Returns:
        V (cupy.ndarray): 3D array of potential values over the grid defined by x, y, and z.
        """
        print("v0:", self.v0)
        x = cp.linspace(-self.extent, self.extent, self.N, dtype=cp.float32)
        y = cp.linspace(-self.extent, self.extent, self.N, dtype=cp.float32)
        z = cp.linspace(-self.extent, self.extent, self.N, dtype=cp.float32)

        # Create a 3D meshgrid for coordinates
        X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

        # Calculate the radial distance from the origin at each point
        r = cp.sqrt(X ** 2 + Y ** 2 + Z ** 2)

        # Compute the potential: apply the condition for truncated potential
        V = cp.where(r >= 1 / self.v0, (1 / r) + self.v0, 0)
        V = cp.reshape(V, (self.N * self.N * self.N, 1), )
        return V


if __name__ == "__main__":
    # Step 3: Create input data_structure
    XLEN, YLEN, ZLEN = 601, 601, 601  # Dimensions of the input arrays and grid
    Z1, Z2 = 1.0, 1.0  # Charges in atomic units
    eps = 1e-10  # Small epsilon to prevent division by zero
    extent = 10.0  # Extent of the grid in Bohr radii (5 Bohr radii in each direction)

    # Coordinates (centered around the nucleus at (0, 0, 0))
    xl = cp.linspace(-extent, extent, XLEN, dtype=cp.float32)
    yl = cp.linspace(-extent, extent, YLEN, dtype=cp.float32)
    zl = cp.linspace(-extent, extent, ZLEN, dtype=cp.float32)


    tex_obj = T.Texture3D(len(xl),len(yl),len(zl))
    sur_obj = S.Surface(len(xl),len(yl),len(zl))
    init_tex = tex_obj.ones()
    init_sur = sur_obj.initial_surface()
    # Step 4: Launch the kernel
    blocksize = (8, 8, 8)  # Assuming BLOCKSIZE in the kernel is 8
    gridsize = (int((XLEN + blocksize[0] - 1) / blocksize[0]),
                int((YLEN + blocksize[1] - 1) / blocksize[1]),
                int((ZLEN + blocksize[2] - 1) / blocksize[2]))

    potential = Potential(x_linspace=xl, Q1=Z1, Q2=Z2, extent=extent)
    potential.operate_cupy(init_tex, init_sur, xl, yl, zl)


    # Step 5: Check the results
    result = sur_obj.get_data(init_sur)  # Move the result back to host (CPU) for inspection
    result_cpu = cp.asnumpy(result)
    # Print the result for inspection

    plt.figure(figsize=(64, 64))
    plt.imshow(result_cpu[300], cmap='viridis', origin='lower')
    plt.colorbar(label='Output value')
    plt.title(f'Output data result 2D summed on x axis')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    plt.figure()
    plt.plot(result_cpu[300][300])
    plt.title(f'Output 1D summed')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    print(result_cpu.sum(axis=0).sum(axis=0).sum(axis=0))
    potential_vector = cp.asnumpy(potential.truncated_potential_3d())
    potential_vector = potential_vector.reshape((XLEN, YLEN, ZLEN))
    potential_vector = potential_vector - potential.v0
    plt.figure(figsize=(32, 32))
    plt.imshow(potential[300], cmap='viridis', origin='lower')
    cbar = plt.colorbar(label='Output value', format='%.2f')
    cbar.ax.tick_params(labelsize=50)
    plt.title(f'Output data result 2D summed on x axis')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    plt.figure()
    plt.plot(potential_vector[300][300])
    plt.title(f'Output 1D summed')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()
