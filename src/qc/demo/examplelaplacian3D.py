#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def main():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    end.record()
    print(start.elapsed_time(end))
    start.record()
    x = torch.randn(1024, 1024, 1024, dtype=torch.float32, device=device0)
    y = torch.randn(1024, 1024, 1024, dtype=torch.float32, device=device0)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    #x = x.to(device0)
    #y = y.to(device0)
    start.record()
    z = x + y
    v = x + y
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    start.record()
    z_cpu = z.to("cpu", torch.float32)
    end.record()
    print(start.elapsed_time(end))

if __name__ == "__main__":
    main()