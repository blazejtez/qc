#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''experimentation with cupy'''
import cupy as cp
#  import numpy as np
#  import time
XLEN = 1000
YLEN = 6
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)
start_event = cp.cuda.stream.Event()
stop_event = cp.cuda.stream.Event()
stream = cp.cuda.stream.Stream()
with stream:
    start_event.record()
    x = cp.random.randn(XLEN**3,YLEN,dtype=cp.float32)
    z = cp.transpose(x).dot(x)*XLEN**-3.
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
