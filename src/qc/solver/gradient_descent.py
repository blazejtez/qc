import time

import numpy as np
import cupy as cp
from qc.hamiltonian.hamiltonian import HamiltonianOperatorCuPy

ALPHA = 0.28294212105225837470023780155114

class GoalGradient():
    def __init__(self, hamiltonian, x):
        self.hamiltonian = hamiltonian
        self.x = x
        self.xtAx_cached = None
        self.xtx_cached = None

    def xtAx(self, x, A):
        # Use cached value if applicable
        if x is self.x and A is self.hamiltonian and self.xtAx_cached is not None:
            return self.xtAx_cached
        # Compute and cache otherwise
        self.xtAx_cached = x.T.dot(A.matvec(x))
        return self.xtAx_cached

    def xtx(self, x):
        # Use cached value if applicable
        if x is self.x and self.xtx_cached is not None:
            return self.xtx_cached
        # Compute and cache otherwise
        self.xtx_cached = x.T.dot(x)
        return self.xtx_cached

    def objective_function(self, x, A):
        return self.xtAx(x, A) / self.xtx(x)

    def gradient(self, x, A):
        # Calculate components of the gradient
        num = 2 * A.matvec(x)
        denom = self.xtx(x)
        xtAx_value = self.xtAx(x, A)
        return (num / denom) - (2 * xtAx_value * x / denom ** 2)


# Define the grid
N = 300
L = 10.0  # Define the size of the grid (in atomic units, for example)

x = np.linspace(-L, L, N, dtype=cp.float32)
y = np.linspace(-L, L, N, dtype=cp.float32)
z = np.linspace(-L, L, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(cp.array(x), cp.array(y), cp.array(z), indexing='ij')

# Compute the Gaussian-type orbital values
gaussian_orbital = cp.exp(-ALPHA*(X**2 + Y**2 + Z**2), dtype=cp.float32)
x0 = cp.reshape(gaussian_orbital, (len(x) * len(y) * len(z), 1))

# Initialize matrix A
A = HamiltonianOperatorCuPy(x, y, z, L)

# Create an instance of the GoalGradient class
goal_gradient = GoalGradient(A, x0)

# Gradient descent using the GoalGradient class
def gradient_descent_cached(goal_gradient, x0, lr=0.0001, tol=1e-6, max_iter=100000):
    x = x0
    A = goal_gradient.hamiltonian
    for i in range(max_iter):
        grad = goal_gradient.gradient(x, A)
        x_new = x - lr * grad

        # Check convergence
        if cp.linalg.norm(x_new - x) < tol:
            break
        print(f"Iteration: {i}, Eigenvalue: {goal_gradient.objective_function(x, A)}, Norm: {cp.linalg.norm(x_new - x)}")
        x = x_new

    return x

# Perform minimization using the cached class
start_time = time.time()
result = gradient_descent_cached(goal_gradient, x0, lr=0.9)
end_time = time.time()

# Final minimized objective function value
min_value = goal_gradient.objective_function(result, A)

print("Zminimalizowane x:", result)
print("Minimalna wartość funkcji:", min_value)
print("Czas wykonania:", end_time - start_time, "sekund")
