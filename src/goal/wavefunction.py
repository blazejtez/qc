import cupy as cp
def hydrogen_1s(N):
    x = cp.linspace(-10, 10, N)
    y = cp.linspace(-10, 10, N)
    z = cp.linspace(-10, 10, N)
    X, Y, Z_grid = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z_grid**2)
    return cp.ravel(cp.exp(-r))

def hydrogen_2s(N):
    x = cp.linspace(-10, 10, N)
    y = cp.linspace(-10, 10, N)
    z = cp.linspace(-10, 10, N)
    X, Y, Z_grid = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z_grid**2)
    return cp.ravel(cp.exp(-r / 2) * (2 - r))

def hydrogen_2px(N):
    x = cp.linspace(-10, 10, N)
    y = cp.linspace(-10, 10, N)
    z = cp.linspace(-10, 10, N)
    X, Y, Z_grid = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z_grid**2)
    return cp.ravel(cp.exp(-r / 2) * X)

# Similarly, for other wavefunctions:

def hydrogen_3s(N):
    x = cp.linspace(-10, 10, N)
    y = cp.linspace(-10, 10, N)
    z = cp.linspace(-10, 10, N)
    X, Y, Z_grid = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z_grid**2)
    return cp.ravel(cp.exp(-r / 3) *
                    ((27 - 18 * r) + 2 * r**2))

def hydrogen_3px(N):
    x = cp.linspace(-10, 10, N)
    y = cp.linspace(-10, 10, N)
    z = cp.linspace(-10, 10, N)
    X, Y, Z_grid = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z_grid**2)
    return cp.ravel(cp.exp(-r / 3) * (6 - r) * X)

def hydrogen_3d_3z2_r2(N):
    x = cp.linspace(-10, 10, N)
    y = cp.linspace(-10, 10, N)
    z = cp.linspace(-10, 10, N)
    X, Y, Z_grid = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z_grid**2)
    return cp.ravel(cp.exp(-r / 3) * (3 * Z_grid**2 - r**2))



def main():
    # Example Usage:
    N = 300  # Grid size (N^3 points)
    Z = 1   # Atomic number (Hydrogen)
    a0 = 1  # Bohr radius in AU
    N_1s = 1  # Normalization constant for 1s

    # wavefunction_1s = hydrogen_1s(N, Z, a0, N_1s)
    # print(wavefunction_1s.shape)  # Should be (N^3,
    # Phi = load_basic("h100.cub").get()

    Phi = hydrogen_2px(N, Z, a0, N_1s).get()

# Other 3d wavefunctions follow a similar pattern.
if __name__ == "__main__":
    main()