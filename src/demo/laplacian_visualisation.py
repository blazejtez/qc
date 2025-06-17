import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

def laplacian_gto(alpha_F, x, y, z):
    """Computes the Laplacian of the GTO function."""
    r_squared = x ** 2 + y ** 2 + z ** 2
    laplacian = 2 * np.exp(-alpha_F * r_squared) * alpha_F * (-3 + 2 * alpha_F * r_squared)
    return laplacian


def plot_laplacian_gto(alpha_F=1.0, grid_size=101, extent=5):
    """Plots the laplacian of GTO basis function, 3 color plots and 1 1D plot."""
    # Create 3D grid
    x = cp.linspace(-extent, extent, grid_size)
    y = cp.linspace(-extent, extent, grid_size)
    z = cp.linspace(-extent, extent, grid_size)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

    x1 = np.linspace(-extent, extent, grid_size)
    y1 = np.linspace(-extent, extent, grid_size)
    z1 = np.linspace(-extent, extent, grid_size)
    X1, Y1, Z1 = np.meshgrid(x, y, z, indexing='ij')


    # Compute the laplacian function on this grid
    laplacian = laplacian_gto(alpha_F, X, Y, Z)

    # Sum along the z-axis and plot X-Y plane
    sum_z_gpu = cp.sum(laplacian, axis=2)
    sum_z = sum_z_gpu.get()
    # Sum along the y-axis and plot X-Z plane
    sum_y_gpu = cp.sum(laplacian, axis=1)
    sum_y = sum_y_gpu.get()
    # Sum along the x-axis and plot Y-Z plane
    sum_x_gpu = cp.sum(laplacian, axis=0)
    sum_x = sum_x_gpu.get()
    # Sum along both y and z axes to create a 1D plot along x-axis
    sum_yz_gpu = np.sum(np.sum(laplacian, axis=1), axis=1)
    sum_yz = sum_yz_gpu.get()
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # X-Y plane
    c1 = axs[0, 0].imshow(sum_z, extent=[-extent, extent, -extent, extent], cmap='plasma')
    axs[0, 0].set_title('Summed along Z-axis (X-Y plane)')
    fig.colorbar(c1, ax=axs[0, 0])

    # X-Z plane
    c2 = axs[0, 1].imshow(sum_y.T, extent=[-extent, extent, -extent, extent], cmap='viridis')
    axs[0, 1].set_title('Summed along Y-axis (X-Z plane)')
    fig.colorbar(c2, ax=axs[0, 1])

    # Y-Z plane
    c3 = axs[1, 0].imshow(sum_x.T, extent=[-extent, extent, -extent, extent], cmap='inferno')
    axs[1, 0].set_title('Summed along X-axis (Y-Z plane)')
    fig.colorbar(c3, ax=axs[1, 0])

    # 1D plot
    axs[1, 1].plot(x1, sum_yz, color='b')
    axs[1, 1].set_title('Summed along Y and Z axes (1D slice along X)')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Summed Laplacian')

    plt.tight_layout()
    plt.show()
    return laplacian

if __name__ == '__main__':
    for i in range(5,15,3):
    # Call the function to plot
        plot_laplacian_gto(alpha_F=0.28, grid_size=100, extent=i)
