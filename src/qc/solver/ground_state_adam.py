#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
import portion as P

import Praktyki.cut_box_3D as box
import qc.hamiltonian.hamiltonian as H
import qc.data_structure.raster as raster
from rayleigh_quotient import GoalGradient


def adam(raster_density=10, max_iters=5000, beta1=0.9, beta2=0.999, epsilon=1e-10, learning_rate=1e-5, tol=1e-10):
    HYDROGEN_RADIUS = 10.
    ALPHA = 0.282942121052

    interval = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
    box_ = box.box3D(interval, interval, interval)
    r = raster.Raster(raster_density)
    xl, yl, zl = r.box_linspaces(box_)
    N = len(xl) * len(yl) * len(zl)

    h = H.HamiltonianOperatorCuPy(xl, yl, zl, extent=HYDROGEN_RADIUS)

    X, Y, Z = cp.meshgrid(cp.array(xl), cp.array(yl), cp.array(zl))
    gaussian_orbital = cp.exp(-ALPHA * (X ** 2 + Y ** 2 + Z ** 2), dtype=cp.float32)
    norm_factor = cp.sum(gaussian_orbital ** 2) * h.h ** 3
    gaussian_orbital /= cp.sqrt(norm_factor)

    v_init = cp.reshape(gaussian_orbital, (len(X) * len(Y) * len(Z), 1))
    goal_gradient = GoalGradient(hamiltonian=h, x=v_init)
    # Initialize first moment (m) and second moment (v)
    m = cp.zeros_like(v_init)
    v = cp.zeros_like(v_init)

    # Initialize time step
    t = 0
    # Optimization loop with Adam
    v_curr = v_init.copy()
    for i in range(max_iters):
        grad, rayleigh_quotient_value = goal_gradient.gradient(v_curr, h)

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
        new_grad, new_rayleigh_quotient_value = goal_gradient.gradient(v_new, h)
        diff = new_rayleigh_quotient_value - rayleigh_quotient_value
        # Check for convergence
        residual_norm = cp.linalg.norm(v_new - v_curr)
        if residual_norm < tol:
            break
        if abs(diff) < tol:
            print("Diff low enough:", diff)
            break


        # Print iteration details
        print("Iteration:", i, "Eigenvalue:", rayleigh_quotient_value, "Residual norm:", residual_norm)

        # Update the current vector
        v_curr = v_new

    # Calculate the final eigenvalue
    eigenvalue = goal_gradient.objective_function(v_curr, h)
    # Print the calculated eigenvalue
    print("Calculated Eigenvalue:", eigenvalue)
    return eigenvalue, v_curr


if __name__ == "__main__":
    adam()

