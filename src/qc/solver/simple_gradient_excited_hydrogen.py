import time

import numpy as np
import cupy as cp

from qc.hamiltonian.hamiltonian import HamiltonianOperatorCuPy

ALPHA = 0.28294212105225837470023780155114

class GoalGradient():
    def __init__(self, hamiltonian, x, Y=None):
        self.hamiltonian = hamiltonian
        self.x = x
        self.Y = Y  # Matrix of previously found eigenvectors
        self.xtAx_cached = None
        self.xtx_cached = None

    def xtAx(self, x, A):
        if x is self.x and A is self.hamiltonian and self.xtAx_cached is not None:
            return self.xtAx_cached
        self.xtAx_cached = x.T.dot(A.matvec(x))
        return self.xtAx_cached

    def xtx(self, x):
        if x is self.x and self.xtx_cached is not None:
            return self.xtx_cached
        self.xtx_cached = x.T.dot(x)
        return self.xtx_cached

    def objective_function(self, x, A, lambd):
        if self.Y is not None and lambd is not None:
            return self.xtAx(x, A) / self.xtx(x) + lambd.T.dot(self.Y.T.dot(x))
        else:
            return self.xtAx(x, A) / self.xtx(x)

    def gradient_x(self, x, A, lambd):
        num = 2 * A.matvec(x)
        denom = self.xtx(x)
        xtAx_value = self.xtAx(x, A)
        if self.Y is not None:
            gradient = (num / denom) - (2 * xtAx_value * x / denom ** 2) + self.Y.dot(lambd)
        else:
            gradient = (num / denom) - (2 * xtAx_value * x / denom ** 2)
        return gradient

    def gradient_lambda(self, x):
        if self.Y is not None:
            return self.Y.T.dot(x)
def gradient_descent_constrained(goal_gradient, x0, lambd0, lr=0.0001, tol=2.3e-5, max_iter=1000000000):
    x = x0
    lambd = lambd0
    A = goal_gradient.hamiltonian
    for i in range(max_iter):
        grad_x = goal_gradient.gradient_x(x, A, lambd)
        grad_lambda = goal_gradient.gradient_lambda(x)

        # Update x and lambda
        x_new = x - lr * grad_x
        if grad_lambda is not None:
            lambd_new = lambd - lr * grad_lambda
        else:
            lambd_new = lambd

        # Normalize x_new to prevent it from becoming too small or large
        x_new = x_new / cp.linalg.norm(x_new)

        # Check convergence
        if cp.linalg.norm(x_new - x) < tol and cp.linalg.norm(lambd_new - lambd) < tol:
            break

        # Optionally, print progress
        eigenvalue = goal_gradient.objective_function(x_new, A, lambd_new)
        print(f"Iteration: {i}, Eigenvalue: {eigenvalue}, Norm change: {cp.linalg.norm(x_new - x)}")

        x = x_new
        lambd = lambd_new

    return x, lambd

def find_lowest_eigenvalues(A, initial_x, num_eigenvalues=4, lr=0.0001):
    eigenvectors = []
    eigenvalues = []
    x0 = initial_x

    for i in range(num_eigenvalues):
        # Initialize lambda (Lagrange multipliers) as zeros
        lambd0 = cp.zeros((len(eigenvectors), 1), dtype=cp.float32) if eigenvectors else cp.zeros((0, 1), dtype=cp.float32)
        Y = cp.hstack(eigenvectors) if eigenvectors else None

        # Create GoalGradient instance with the current Y
        goal_gradient = GoalGradient(A, x0, Y=Y)

        # Perform constrained gradient descent
        x, lambd = gradient_descent_constrained(goal_gradient, x0, lambd0, lr=lr)

        # Compute the eigenvalue
        eigenvalue = goal_gradient.objective_function(x, A, lambd)

        # Store the eigenvector and eigenvalue
        eigenvectors.append(x)
        eigenvalues.append(eigenvalue)

        # Prepare for next iteration
        x0 = cp.random.rand(*x.shape).astype(cp.float32)
        x0 = x0 - sum([vec.dot(vec.T.dot(x0)) for vec in eigenvectors])  # Orthogonalize initial guess

    return eigenvalues, eigenvectors


# Define the grid
N = 201  # Reduced for computational feasibility
L = 10.0

x = np.linspace(-L, L, N, dtype=cp.float32)
y = np.linspace(-L, L, N, dtype=cp.float32)
z = np.linspace(-L, L, N, dtype=cp.float32)
sh = N**3
X, Y, Z = cp.meshgrid(cp.array(x), cp.array(y), cp.array(z), indexing='ij')

# Compute the Gaussian-type orbital values
gaussian_orbital = cp.exp(-ALPHA*(X**2 + Y**2 + Z**2), dtype=cp.float32)
x0 = cp.reshape(gaussian_orbital, (len(x) * len(y) * len(z), 1))

# Y = load_cub("h100.cub")
# Y = cp.reshape(Y, (sh, 1), )
# lamd = cp.asarray([[-0.49654406]], dtype=cp.float32)
Y = None
lamd = None

# Initialize matrix A
A = HamiltonianOperatorCuPy(x, y, z, L)

# Find the lowest five eigenvalues
start_time = time.time()
eigenvalues, eigenvectors = find_lowest_eigenvalues(A, x0, num_eigenvalues=5, lr=0.0001)
end_time = time.time()

# Display the results
for idx, eigenvalue in enumerate(eigenvalues):
    print(f"Eigenvalue {idx+1}: {eigenvalue}")

print("Czas wykonania:", end_time - start_time, "sekund")
