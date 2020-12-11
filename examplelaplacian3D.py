#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
tic = time.perf_counter()
a = np.random.randn(1024,1024,1024).astype(np.float32)
toc = time.perf_counter()

print(f"Generation time: {toc-tic:4f} seconds") 
tic = time.perf_counter()
a += 8*a
toc = time.perf_counter()
print(f"Addition time: {toc-tic:4f} seconds") 

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
end.record()
print(start.elapsed_time(end))
start.record()
x = torch.randn(1024, 1024, 1024, dtype=torch.float32, device=device0)
#y = torch.randn(1024, 1024, 1024, dtype=torch.float32, device=device0)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))
#x = x.to(device0)
#y = y.to(device0)
start.record()
x += 8*x 
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))
start.record()
x_cpu = x.to("cpu", torch.float32)
end.record()
print(start.elapsed_time(end))
