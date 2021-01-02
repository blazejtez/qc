#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''experimentation with cupy'''
#  import time
import cupy as cp
XLEN = 1000
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
    channel_descriptor = cp.cuda.texture.ChannelFormatDescriptor(
        NUMBITS, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
    cuda_array = cp.cuda.texture.CUDAarray(channel_descriptor, XLEN, YLEN,
                                           ZLEN)
    x_resaped = cp.reshape(x,(XLEN,YLEN,ZLEN))
    cuda_array.copy_from(x_resaped)
    resource_descriptor = cp.cuda.texture.ResourceDescriptor(
        cp.cuda.runtime.cudaResourceTypeArray, cuda_array)
    surface_obj = cp.cuda.texture.SurfaceObject(resource_descriptor)
    z = cp.transpose(x).dot(x) * XLEN**-3.
    stop_event.record()
    stop_event.synchronize()
    print(z)
    print(f"{cp.cuda.get_elapsed_time(start_event,stop_event)*1e-3}")
#  tic = time.time()
#  x_cpu = np.random.randn(xlen**3,ylen).astype(np.float32)
#  z_cpu = np.transpose(x_cpu).dot(x_cpu)*xlen**-3.
#  print(z_cpu)
#  toc = time.time()
#  print(f"{toc-tic}")
