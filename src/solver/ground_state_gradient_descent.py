#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cupy as cp
import portion as P
import time

import Praktyki.cut_box_3D as box
import hamiltonian.hamiltonian as H
import data_structure.raster as raster
from rayleigh_quotient import GoalGradient

# Gradient descent using the GoalGradient class
def gradient_descent(raster_density, lr=0.001, tol=1e-6, max_iter=100000):
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
    x = v_init
    A = goal_gradient.hamiltonian
    for i in range(max_iter):
        grad, eigenvalue = goal_gradient.gradient(x, A)
        x_new = x - lr * grad

        # Check convergence
        if cp.linalg.norm(x_new - x) < tol:
            break
        print(f"Iteration: {i}, Eigenvalue: {goal_gradient.objective_function(x, A)}, Norm: {cp.linalg.norm(x_new - x)}")
        x = x_new

    return x, goal_gradient

def main():
    # Perform minimization using the cached class
    start_time = time.time()
    result, goal_gradient = gradient_descent(raster_density=10, lr=1, tol=1e-5)
    end_time = time.time()

    # Final minimized objective function value
    min_value = goal_gradient.objective_function(result, goal_gradient.hamiltonian)

    print("Zminimalizowane x:", result)
    print("Minimalna wartość funkcji:", min_value)
    print("Czas wykonania:", end_time - start_time, "sekund")

if __name__ == '__main__':
    main()

