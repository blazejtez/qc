#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
import portion as P

import Praktyki.cut_box_3D as box
import qc.hamiltonian.hamiltonian as H
import qc.data.raster as raster

HYDROGEN_RADIUS = 10.
ALPHA = 0.28294212105225837470023780155114

interval = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
box_ = box.box3D(interval, interval, interval)
r = raster.Raster(10)
xl, yl, zl = r.box_linspaces(box_)
N = len(xl) * len(yl) * len(zl)
print("N:", N)

h = H.HamiltonianOperatorCuPy(xl, yl, zl, extent=HYDROGEN_RADIUS)

# tutaj początkowym wektorem jest losowy wektor znormalizowany. To dajmy funkcję GTO w to miejsce.
# v_init = cp.random.randn(*(len(xl) * len(yl) * len(zl), 1), dtype=cp.float32)
#
# v_init = v_init / cp.linalg.norm(v_init)

X, Y, Z = cp.meshgrid(cp.array(xl), cp.array(yl), cp.array(zl))
gaussian_orbital = cp.exp(-ALPHA*(X**2 + Y**2 + Z**2), dtype=cp.float32)
v_init = cp.reshape(gaussian_orbital, (len(X) * len(Y) * len(Z), 1))




# Adam parameters
beta1 = 0.9  # Exponential decay rate for the first moment
beta2 = 0.999  # Exponential decay rate for the second moment
epsilon = 1e-10  # Small constant to prevent division by zero
learning_rate = 1e-5  # Step size
max_iters = 10000
tolerance = 1e-4

# Initialize first moment (m) and second moment (v)
m = cp.zeros_like(v_init)
v = cp.zeros_like(v_init)

# Initialize time step
t = 0


# Gradient Descent to Minimize Rayleigh Quotient with Adam
def rayleigh_quotient(v, h):
    v = v.reshape(-1, 1)  # Ensure v is (size, 1)
    num = cp.dot(v.T, h.matvec(v))  # Numerator: v^H * H * v
    denom = cp.dot(v.T, v)  # Denominator: v^H * v
    return cp.real(num / denom)


def gradient(v, h):
    v = v.reshape(-1, 1)  # Ensure v is (size, 1)
    Hv = h.matvec(v)  # H * v
    denom = cp.dot(v.T, v)
    rayleigh_quotient_value = rayleigh_quotient(v, h)
    grad = (Hv - rayleigh_quotient_value * v) / denom  # Gradient of Rayleigh quotient
    return grad, rayleigh_quotient_value

if __name__ == "__main__":
    # Optimization loop with Adam
    v_curr = v_init.copy()

    for i in range(max_iters):
        grad, rayleigh_quotient_value = gradient(v_curr, h)

        # Adam update
        t += 1
        m = beta1 * m + (1 - beta1) * grad  # Update biased first moment estimate
        v = beta2 * v + (1 - beta2) * cp.square(grad)  # Update biased second moment estimate

        # Correct bias in first and second moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update variables
        v_new = v_curr - learning_rate * m_hat / (cp.sqrt(v_hat) + epsilon)

        # Normalize the new vector
        v_new /= cp.linalg.norm(v_new)

        # Check for convergence
        residual_norm = cp.linalg.norm(v_new - v_curr)
        if residual_norm < tolerance:
            break

        # Print iteration details
        print("Iteration:", i, "Eigenvalue:", rayleigh_quotient_value, "Residual norm:", residual_norm)

        # Update the current vector
        v_curr = v_new


# Calculate the final eigenvalue
eigenvalue = rayleigh_quotient(v_curr, h)

# Print the calculated eigenvalue
print("Calculated Eigenvalue:", eigenvalue)
