import cupy as cp
def hydrogen_1s(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h1s = cp.ravel(cp.exp(-r)).reshape(N**3, 1)
    h1s /= cp.linalg.norm(h1s)
    return h1s

def hydrogen_2s(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h2s = cp.ravel(cp.exp(-r/2) * (2 - r)).reshape(N**3, 1)
    h2s /= cp.linalg.norm(h2s)
    return h2s

def hydrogen_2px(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h2px = cp.ravel(cp.exp(-r / 2) * X).reshape(N ** 3, 1)
    h2px /= cp.linalg.norm(h2px)
    return h2px

def hydrogen_2py(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h2py = cp.ravel(cp.exp(-r / 2) * Y).reshape(N ** 3, 1)
    h2py /= cp.linalg.norm(h2py)
    return h2py

def hydrogen_2pz(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h2pz = cp.ravel(cp.exp(-r / 2) * Y).reshape(N ** 3, 1)
    h2pz /= cp.linalg.norm(h2pz)
    return h2pz

def hydrogen_3s(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h3s = cp.ravel(cp.exp(-r)).reshape(N**3, 1)
    h3s /= cp.linalg.norm(h3s)
    return h3s

def hydrogen_3px(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h3px = cp.ravel(cp.exp(-r / 3) * (6 - r) * X).reshape(N ** 3, 1)
    h3px /= cp.linalg.norm(h3px)
    return h3px

def hydrogen_3py(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h2py = cp.ravel(cp.exp(-r / 3) * (6 - r) * Y).reshape(N ** 3, 1)
    h2py /= cp.linalg.norm(h2py)
    return h2py

def hydrogen_3pz(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h2pz = cp.ravel(cp.exp(-r / 3) * (6 - r) * Z).reshape(N ** 3, 1)
    return h2pz

def hydrogen_3d_3z2_r2(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z**2)
    h3d = cp.ravel(cp.exp(-r / 3) * (3 * Z**2 - r**2)).reshape(N ** 3, 1)
    h3d /= cp.linalg.norm(h3d)
    return h3d