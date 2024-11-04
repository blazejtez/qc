import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import lobpcg

from hamiltonian.hamiltonian import HamiltonianOperatorCuPy

ALPHA = 0.28294212105225837470023780155114

# Define the grid
N = 201
L = 15.0

x = np.linspace(-L, L, N, dtype=cp.float32)
y = np.linspace(-L, L, N, dtype=cp.float32)
z = np.linspace(-L, L, N, dtype=cp.float32)
sh = N**3
X, Y, Z = cp.meshgrid(cp.array(x), cp.array(y), cp.array(z), indexing='ij')

# Compute the Gaussian-type orbital values
gaussian_orbital = cp.exp(-ALPHA*(X**2 + Y**2 + Z**2), dtype=np.float32)
x0 = cp.reshape(gaussian_orbital, (len(x) * len(y) * len(z), 1))
x0.astype(cp.float32)
x01 = cp.random.random((sh,5), dtype=np.float32)
x0 = cp.hstack((x0, x01))

# Initialize matrix A
A = HamiltonianOperatorCuPy(x, y, z, L)

w, v = lobpcg(A, x0, largest=False, verbosityLevel=3)
