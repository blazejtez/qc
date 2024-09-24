#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
import portion as P
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lobpcg as lp

import Praktyki.cut_box_3D as box
import qc.hamiltonian.hamiltonian as H
import qc.data.raster as raster

HYDROGEN_RADIUS = 10.

intvl = P.closed(-HYDROGEN_RADIUS,HYDROGEN_RADIUS)
box_ = box.box3D(intvl,intvl,intvl)
r = raster.Raster(10)
xl, yl, zl = r.box_linspaces(box_)
N = len(xl)*len(yl)*len(zl)

h = H.HamiltonianOperatorCuPy(xl,yl,zl)

v_init = cp.random.randn(*(len(xl)*len(yl)*len(zl),1),dtype=cp.float32)

v_init = v_init/np.linalg.norm(v_init)

# here magic begins

# Gradient Descent to Minimize Rayleigh Quotient
def rayleigh_quotient(v, h):
    v = v.flatten()  # Ensure v is 1D
    num = cp.dot(v.T, h.matvec(v))  # Numerator: v^H * H * v
    denom = cp.dot(v.T, v)           # Denominator: v^H * v
    return cp.real(num / denom)

def gradient(v, h):
    v = v.flatten()
    Hv = h.matvec(v)  # H * v
    denom = cp.dot(v.T, v)
    rayleigh_quotient_value = rayleigh_quotient(v, h)
    grad = (Hv - rayleigh_quotient_value * v) / denom  # Gradient of Rayleigh quotient
    return grad, rayleigh_quotient_value

# Parameters for Gradient Descent
learning_rate = 0.1
max_iters = 10
tolerance = 1e-6

v = v_init.copy()
for i in range(max_iters):
    grad, rayleigh_quotient_value = gradient(v, h)
    print("Iteration:", i, "Eigenvalue:", rayleigh_quotient_value)
    v_new = (v.T - learning_rate * grad).T  # Update step
    v_new /= cp.linalg.norm(v_new)  # Normalize

    if cp.linalg.norm(v_new - v) < tolerance:  # Check for convergence
        break
    v = v_new

# Calculate the final eigenvalue
eigenvalue = rayleigh_quotient(v, h)

# Print the calculated eigenvalue
print("Calculated Eigenvalue:", eigenvalue)