import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import lobpcg, LinearOperator

from hamiltonian.hamiltonian import HamiltonianOperatorCuPy
from goal.wavefunction import hydrogen_1s, hydrogen_2s, hydrogen_2px, hydrogen_3px, hydrogen_3s, hydrogen_3d_3z2_r2

ALPHA = np.float32(0.28294212105225837470023780155114)

# Define the grid
N = 201
L = 15.0

x = np.linspace(-L, L, N, dtype=cp.float32)
y = np.linspace(-L, L, N, dtype=cp.float32)
z = np.linspace(-L, L, N, dtype=cp.float32)
sh = N**3
X, Y, Z = cp.meshgrid(cp.array(x), cp.array(y), cp.array(z), indexing='ij')



h1s = hydrogen_1s(N)
h2s = hydrogen_2s(N)
h2p = hydrogen_2px(N)
h3s = hydrogen_3s(N)
h3px = hydrogen_3px(N)
h3d = hydrogen_3d_3z2_r2(N)

h1s = h1s/cp.linalg.norm(h1s)
h2s = h2s/cp.linalg.norm(h2s)
h2p = h2p/cp.linalg.norm(h2p)
h3s = h3s/cp.linalg.norm(h3s)
h3px = h3px/cp.linalg.norm(h3px)
h3d = h3d/cp.linalg.norm(h3d)

vec = cp.column_stack((h1s, h2s, h2p, h3s, h3px, h3d))

vec = cp.array(vec)

# rng = np.random.default_rng()
# vec = rng.uniform(size=(sh, 10))
# vec = vec / np.linalg.norm(vec)
# vec = vec.astype(cp.float32)
# vec = cp.asarray(vec)

# Initialize matrix A
hamiltonian = HamiltonianOperatorCuPy(x, y, z, L)
A = LinearOperator(shape=hamiltonian.shape,
                   matvec=hamiltonian.matvec,
                   matmat=hamiltonian.matmat,
                   dtype=hamiltonian.dtype)

w, v = lobpcg(A, vec, maxiter=1000000, largest=False, verbosityLevel=3)
