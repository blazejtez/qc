import cupy as cp
import math

# Define constants
alpha = 0.28294212105225837470023780155114  # GTO parameter, controls width of Gaussian
pi = math.pi

# Define grid for evaluation
N = 600  # Number of grid points per dimension
L = 20.0  # Size of the grid box (in atomic units)
dx = L / N  # Grid spacing

# Create 3D grid of points (x, y, z) for GPU
x = cp.linspace(-L/2, L/2, N)
y = cp.linspace(-L/2, L/2, N)
z = cp.linspace(-L/2, L/2, N)
X, Y, Z = cp.meshgrid(x, y, z)

# Calculate radial distance r from the origin
R = cp.sqrt(X**2 + Y**2 + Z**2)

# Define Gaussian wavefunction ψ(r) = exp(-alpha * r^2)
psi = cp.exp(-alpha * R**2)

# Normalize the wavefunction
norm_factor = cp.sum(psi**2) * dx**3
psi /= cp.sqrt(norm_factor)

# Kinetic energy operator: -1/2 * ∇^2 (Laplace operator in Cartesian coordinates)
# Laplace operator in 3D for ψ(r)
laplacian_psi = (
    (cp.roll(psi, 1, axis=0) + cp.roll(psi, -1, axis=0) - 2 * psi) / dx**2 +
    (cp.roll(psi, 1, axis=1) + cp.roll(psi, -1, axis=1) - 2 * psi) / dx**2 +
    (cp.roll(psi, 1, axis=2) + cp.roll(psi, -1, axis=2) - 2 * psi) / dx**2
)
for i in range(-51,50,5):
    V0 = i
    # Kinetic energy expectation value ⟨T⟩ = -1/2 ∫ψ*(-∇²ψ) dV
    kinetic_energy = -0.5 * cp.sum(psi * laplacian_psi) * dx**3

    # Potential energy operator: V(r)
    potential_operator = cp.where(R > 1/V0, 1 / R + V0, 0)  # Modify potential based on V0

    # Potential energy expectation value ⟨V⟩ = ∫ψ*(Vψ) dV
    potential_energy = cp.sum(psi * potential_operator * psi) * dx**3

    # Total energy expectation value
    total_energy = kinetic_energy + potential_energy

    # Transfer data back to CPU for display
    kinetic_energy = kinetic_energy.get()
    potential_energy = potential_energy.get() - V0
    total_energy = total_energy.get() - V0
    print("V0:", V0)
    print(f"Kinetic Energy (T): {kinetic_energy:.6f} hartree")
    print(f"Potential Energy (V): {potential_energy:.6f} hartree")
    print(f"Total Energy (E): {total_energy:.6f} hartree")
