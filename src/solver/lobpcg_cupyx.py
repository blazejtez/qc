import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import lobpcg, LinearOperator

from hamiltonian.hamiltonian import HamiltonianOperatorCuPy
from goal.wavefunction import hydrogen_1s, hydrogen_2s, hydrogen_2px, hydrogen_2py, hydrogen_2pz, hydrogen_3s, hydrogen_3px, hydrogen_3py, hydrogen_3pz, hydrogen_3d_3z2_r2

ALPHA = np.float32(0.28294212105225837470023780155114)

# Define the grid
N = 301
L = 40.0

x = np.linspace(-L, L, N, dtype=cp.float32)
y = np.linspace(-L, L, N, dtype=cp.float32)
z = np.linspace(-L, L, N, dtype=cp.float32)
sh = N**3
X, Y, Z = cp.meshgrid(cp.array(x), cp.array(y), cp.array(z), indexing='ij')


def init_wf():
    h1s = hydrogen_1s(N)
    h2s = hydrogen_2s(N)
    h2px = hydrogen_2px(N)
    h2py = hydrogen_2py(N)
    h2pz = hydrogen_2pz(N)
    h3s = hydrogen_3s(N)
    #h3px = hydrogen_3px(N)
    #h3py = hydrogen_3py(N)
    #h3pz = hydrogen_3pz(N)
    #h3d = hydrogen_3d_3z2_r2(N)

    h1s = h1s/cp.linalg.norm(h1s)
    h2s = h2s/cp.linalg.norm(h2s)
    h2px = h2px/cp.linalg.norm(h2px)
    h2py = h2py/cp.linalg.norm(h2py)
    h2pz = h2pz/cp.linalg.norm(h2pz)
    h3s = h3s/cp.linalg.norm(h3s)
    #h3px = h3px/cp.linalg.norm(h3px)
    #h3py = h3py/cp.linalg.norm(h3py)
    #h3pz = h3pz/cp.linalg.norm(h3pz)
    #h3d = h3d/cp.linalg.norm(h3d)
    vec = cp.column_stack((h1s, h2s, h2px, h2py, h2pz, h3s))#, h3px, h3py))#, h3d))
    return cp.asarray(vec)

def init_random():
    import time
    time_seed = int(time.time())
    rng = np.random.default_rng(time_seed)
    vec = rng.uniform(size=(sh, 12))
    for col_idx in range(vec.shape[1]):
        column = vec[:, col_idx]  # Select the column as a slice
        column /= np.linalg.norm(column)
    vec = vec.astype(cp.float32)
    vec = cp.asarray(vec)

vec = init_wf() # or you can use init_random()
# Initialize matrix A
hamiltonian = HamiltonianOperatorCuPy(x, y, z, L)
A = LinearOperator(shape=hamiltonian.shape,
                   matvec=hamiltonian.matvec,
                   matmat=hamiltonian.matmat,
                   dtype=hamiltonian.dtype)

w, v = lobpcg(A, vec, maxiter=1000, tol=1e-3, largest=False, verbosityLevel=3)
