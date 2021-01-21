#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checking working of Texture and Surface classes"""
import cupy as cp
import numpy as np
import qc.surface as S
import qc.texture as T

test_kernel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN){
    for(unsigned int z = 0; z < ZLEN; z++){
        for(unsigned int y = 0; y < YLEN; y++){
            for(unsigned int x = 0; x < XLEN; x++){
               float value = tex3D<float>(texture_input, (float)x, float(y), float(z));
               printf("%f ", value);
               value *= 2;
               surf3Dwrite<float>(value,surface_output,x*sizeof(float),y,z);
            }
        }
    }
    printf("\n");
                                 }''',
    'test',
    backend='nvcc')

test_kernel_parallel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;
    if( (x < XLEN) && (y < YLEN) && (z < ZLEN)){
               float value = tex3D<float>(texture_input, (float)x, float(y), float(z));
               printf("%f ", value);
               value *= 2;
               surf3Dwrite<float>(value,surface_output,x*sizeof(float),y,z);
               }
               if(x == 0 && y == 0 && z == 0)
    printf("\n");
                                 }''',
    'test',
    backend='nvcc')

test_kernel_parallel_full = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN, float h3){
    int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
    int idx_z = threadIdx.z + blockIdx.z*blockDim.z;
    float value = 0.;
    if( (idx_x < XLEN) && (idx_y < YLEN) && (idx_z < ZLEN)){

        value = -tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z)*6 + \
        tex3D<float>(texture_input, (float)idx_x - 1, (float)idx_y, (float)idx_z) + \
        tex3D<float>(texture_input, (float)idx_x + 1, (float)idx_y, (float)idx_z) + \
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y - 1, (float)idx_z) + \
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y + 1, (float)idx_z) + \
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z - 1) + \
        tex3D<float>(texture_input, (float)idx_x, (float)idx_y, (float)idx_z + 1);
               //printf("%e ", value);
               //value /= h3;
               surf3Dwrite<float>(value,surface_output,idx_x*sizeof(float),idx_y,idx_z);
               }
               //  if(idx_x == 0 && idx_y == 0 && idx_z == 0)
    //  printf("\n");
                                 }''',
    'test',
    backend='nvcc')
if __name__ == "__main__":
    XLEN = 1000
    YLEN = ZLEN = XLEN
    tex_obj = T.Texture(XLEN, YLEN, ZLEN)
    sur_obj = S.Surface(XLEN, YLEN, ZLEN)

    init_tex = tex_obj.initial_texture()

    init_sur = sur_obj.initial_surface()

    test_kernel.compile()
    test_kernel_parallel.compile()
    test_kernel_parallel_full.compile()

    start_event = cp.cuda.stream.Event()
    stop_event = cp.cuda.stream.Event()
    stream = cp.cuda.stream.Stream()
    with stream:

        for i in range(3):

            start_event.record()
            #  test_kernel((1, ), (1, ), (
            #      init_tex,
            #      init_sur,
            #      XLEN,
            #      YLEN,
            #      ZLEN,
            #  ))
            #
            h3 = 0.2;
            test_kernel_parallel_full((np.int(np.ceil(1000./64)),np.int(np.ceil(1000./64)),np.int(np.ceil(1000./64))),(64,64,64), (init_tex, init_sur, XLEN, YLEN, ZLEN, h3))

            init_tex = tex_obj.texture_from_surface(init_sur)
            y = sur_obj.get_data(init_sur) 
            stop_event.record()
            stop_event.synchronize()
            print(i)
            #  print(y)
            print(cp.cuda.stream.get_elapsed_time(start_event,stop_event))
