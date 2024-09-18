import time

import numpy as np
import cupy as cp
from qc.hamiltonian.hamiltonian import HamiltonianOperatorCuPy

def xtx(x: cp.ndarray, A) -> cp.float32:
    return x.T.dot(A.matvec(x)) / (x.T.dot(x))

def objective_function(x: cp.ndarray, A) -> cp.float32:
    return xtx(x, A) / (x.T.dot(x))


def gradient(x: cp.ndarray, A):
    num = 2 * (A.matvec(x))
    denom = (x.T @ x)
    return (num / denom) - (2 * x.T.dot(A.matvec(x)) * x / (denom ** 2).T)


# Algorytm gradientu prostego
def gradient_descent(A, x0, lr=0.01, tol=1e-5, max_iter=1000):
    x = x0

    for i in range(max_iter):
        print("norma x:", cp.linalg.norm(x))
        grad = gradient(x, A)
        x_new = x - lr * grad
        # Sprawdzanie zbieżności
        print("Eigenvalue:", xtx(x_new, A))
        print("Objective function:", objective_function(x_new, A))
        if cp.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x


# Define the grid
N = 601
L = 10.0  # Define the size of the grid (in atomic units, for example)

x = cp.linspace(-L, L, N, dtype=cp.float32)
y = cp.linspace(-L, L, N, dtype=cp.float32)
z = cp.linspace(-L, L, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

# Compute the Gaussian-type orbital values
gaussian_orbital = cp.exp(-0.5*(X**2 + Y**2 + Z**2), dtype=cp.float32)
print(gaussian_orbital)
x0 = cp.reshape(gaussian_orbital, (len(x) * len(y) * len(z)))

# Inicjalizacja macierzy A i wektora początkowego x0
xl = cp.linspace(-5, 5, N, dtype=cp.float32)
yl = xl
zl = xl
A = HamiltonianOperatorCuPy(xl, yl, zl)
# Minimalizacja funkcji
result = gradient_descent(A, x0, lr=0.9)
min_value = objective_function(result, A)

print("Zminimalizowane x:", result)
print("Minimalna wartość funkcji:", min_value)
