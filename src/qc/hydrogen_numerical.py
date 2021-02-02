#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
import portion as P
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lobpcg as lp

import Praktyki.cut_box as box
import qc.hydrogenh as H
import qc.raster as raster
import torch

x = torch.lobpcg()

HYDROGEN_RADIUS = 6.

intvl = P.closed(-HYDROGEN_RADIUS,HYDROGEN_RADIUS)
box_ = box.box3D(intvl,intvl,intvl)
r = raster.Raster(50)
xl, yl, zl = r.box_linspaces(box_)
N = len(xl)*len(yl)*len(zl)

h = H.HamiltonianOperatorNumPy(xl,yl,zl)

hs = H.HamiltonianSquaredOperatorNumPy(xl,yl,zl)

l = LinearOperator(matvec = hs.matvec, matmat = hs.matmat, dtype = np.float32, shape = h.shape) 

v_init = cp.random.randn(*(len(xl)*len(yl)*len(zl),1),dtype=cp.float32)

v_init = v_init/np.linalg.norm(v_init)

v_init_numpy = cp.asnumpy(v_init)


lp(l, v_init_numpy, largest=True, verbosityLevel=2, maxiter=500)

