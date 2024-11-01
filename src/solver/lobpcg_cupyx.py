import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import lobpcg, LinearOperator

from hamiltonian.hamiltonian import HamiltonianOperatorCuPy

ALPHA = np.float32(0.28294212105225837470023780155114)

# Define the grid
N = 201
L = 15.0

x = np.linspace(-L, L, N, dtype=cp.float32)
y = np.linspace(-L, L, N, dtype=cp.float32)
z = np.linspace(-L, L, N, dtype=cp.float32)
sh = N**3
X, Y, Z = cp.meshgrid(cp.array(x), cp.array(y), cp.array(z), indexing='ij')

rng = np.random.default_rng()
vec = rng.normal(size=(sh, 10))
vec = vec / np.linalg.norm(vec)
vec = vec.astype(cp.float32)
vec = cp.asarray(vec)

# Initialize matrix A
hamiltonian = HamiltonianOperatorCuPy(x, y, z, L)
A = LinearOperator(shape=hamiltonian.shape,
                   matvec=hamiltonian.matvec,
                   matmat=hamiltonian.matmat,
                   dtype=hamiltonian.dtype)

w, v = lobpcg(A, vec, maxiter=1000000, largest=False, verbosityLevel=3)
