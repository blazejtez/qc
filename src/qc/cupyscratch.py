#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''experimentation with cupy'''
import cupy as cp

#  hamiltonian_3d_kernel = cp.RawKernel(
    #  r'''
#  extern "C" __global__ void hamiltonian_3d_kernel(cudaSurfaceObject_t *surface_output, cudaTextureObject_t *texture_input
                                                 #  float *x_linspace, float *y_linspace, float *z_linspace,
                                                 #  int XLEN, int YLEN, int ZLEN) {
#
#  }
#  ''', 'hamiltonian_3d_kernel')
test_kernel_basic = cp.RawKernel(r'''extern "C" __global__ void test(cudaTextureObject_t texture_input, cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN){
    for(unsigned int z = 0; z < ZLEN; z++){
        for(unsigned int y = 0; y < YLEN; y++){
            for(unsigned int x = 0; x < XLEN; x++){
               //printf("hello! %d %d %d ", x, y, z);
               float value = tex3D<float>(texture_input, (float)x, float(y), float(z));
               printf("%f ", value);

               surf3Dwrite<float>(value,surface_output,x*sizeof(float),y,z);
            }
        }
    }
                                 }''', 'test', backend='nvcc')
#  test_kernel = cp.RawKernel(r'''
#  #include <cstdio>
#  __global__ void test_kernel(){ //cudaTextureObject_t texture_input, int XLEN, int YLEN, int ZLEN)
#      for(int x = 0; x < 2; x++){
#          for(int y = 0; y < 2; y++){
#              for(int z = 0; z < 2; z++){
#                 // printf("hello! %d %d %d ", x, y, z);
#                 //float value = tex3D<float>(texture_input, (float)x, float(y), float(z));
#              }
#          }
#      }
#                             }''','test_kernel',options = ('-I/usr/local/cuda/include','-Xcompiler=-fPIC', '-std=c++11'), backend='nvcc')
test_kernel_basic.compile()
#  test_kernel.compile()
XLEN = 2
YLEN = ZLEN = XLEN
NUMVECTORS = 1
NUMBITS = 32
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)
start_event = cp.cuda.stream.Event()
stop_event = cp.cuda.stream.Event()
stream = cp.cuda.stream.Stream()
with stream:
    start_event.record()
    x = cp.random.randn(XLEN * YLEN * ZLEN, NUMVECTORS, dtype=cp.float32)
    y = cp.empty((XLEN*YLEN*ZLEN,1), dtype=cp.float32)
    # create surface output object
    channel_descriptor_input = cp.cuda.texture.ChannelFormatDescriptor(
        NUMBITS, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
    channel_descriptor_output = cp.cuda.texture.ChannelFormatDescriptor(
        NUMBITS, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
    cuda_array_output = cp.cuda.texture.CUDAarray(channel_descriptor_output, XLEN,
                                                  YLEN, ZLEN)
    cuda_array_input = cp.cuda.texture.CUDAarray(channel_descriptor_input, XLEN,
                                                 YLEN, ZLEN)
    x_reshaped = cp.reshape(x, (XLEN, YLEN, ZLEN))
    y_reshaped = cp.reshape(y, (XLEN,YLEN,ZLEN))
    cuda_array_input.copy_from(x_reshaped)
    cuda_array_output.copy_from(y_reshaped)
    resource_descriptor_output = cp.cuda.texture.ResourceDescriptor(
        cp.cuda.runtime.cudaResourceTypeArray, cuda_array_output)
    resource_descriptor_input = cp.cuda.texture.ResourceDescriptor(
        cp.cuda.runtime.cudaResourceTypeArray, cuda_array_input)
    # create surface output object
    surface_obj = cp.cuda.texture.SurfaceObject(resource_descriptor_output)
    # create texture input object
    texture_descriptor = cp.cuda.texture.TextureDescriptor(
        [
            cp.cuda.runtime.cudaAddressModeBorder,
            cp.cuda.runtime.cudaAddressModeBorder,
            cp.cuda.runtime.cudaAddressModeBorder
        ], cp.cuda.runtime.cudaFilterModePoint,
        cp.cuda.runtime.cudaReadModeElementType)
    texture_obj = cp.cuda.texture.TextureObject(resource_descriptor_input,
                                                texture_descriptor)
    # run test kernel
    #  test_kernel((1,),(1,),(texture_obj,XLEN,YLEN,ZLEN))
    test_kernel_basic((1,),(1,),(texture_obj, surface_obj, XLEN,YLEN,ZLEN,))
    y_reshaped = cp.reshape(y,(XLEN,YLEN,ZLEN))
    surface_obj.ResDesc.cuArr.copy_to(y_reshaped)

    #  test_kernel((1,),(1,),(XLEN,YLEN,ZLEN,))
    z = cp.transpose(x).dot(x) * XLEN**-3.
    stop_event.record()
    stop_event.synchronize()
    print(x.T)
    print(y_reshaped)

    print(f"{cp.cuda.get_elapsed_time(start_event,stop_event)*1e-3}")
#  tic = time.time()
#  x_cpu = np.random.randn(xlen**3,ylen).astype(np.float32)
#  z_cpu = np.transpose(x_cpu).dot(x_cpu)*xlen**-3.
#  print(z_cpu)
#  toc = time.time()
#  print(f"{toc-tic}")

