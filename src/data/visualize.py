import numpy as np
from data_structure.util_cub import load_basic
from mayavi.modules.api import ScalarCutPlane
from mayavi import mlab
import cupy as cp
from goal.wavefunction import *

# Example Usage:
N = 300  # Grid size (N^3 points)
Z = 1   # Atomic number (Hydrogen)
a0 = 1  # Bohr radius in AU
N_1s = 1  # Normalization constant for 1s

Phi = load_basic("..\\data\\eigenvector_4.cub").get()

#Phi = hydrogen_2s(N).get()
#normalizing_factor = Phi.T.dot(Phi)
#Phi /= normalizing_factor
#Phi = Phi.reshape((N, N, N))

mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
# We create a scalar field with the module of Phi as the scalar
src = mlab.pipeline.scalar_field(np.abs(Phi))

# And we add the phase of Phi as an additional array
# This is a tricky part: the layout of the new array needs to be the same
# as the existing dataset, and no checks are performed. The shape needs
# to be the same, and so should the data. Failure to do so can result in
# segfaults.
src.image_data.point_data.add_array(np.angle(Phi).T.ravel())
# We need to give a name to our new dataset.
src.image_data.point_data.get_array(1).name = 'angle'
# Make sure that the dataset is up to date with the different arrays:
src.update()
mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', slice_index=50)  # Example: slicing along x
# We select the 'scalar' attribute, ie the norm of Phi
src2 = mlab.pipeline.set_active_attribute(src,
                                    point_scalars='scalar')

# Cut isosurfaces of the norm
contour = mlab.pipeline.contour(src2)
contour.filter.auto_contours = False  # Disable automatic contours
contour.filter.number_of_contours = 1  # Set the number of contours
contour.filter.contours = [7e-5]  # Example contours

# Now we select the 'angle' attribute, ie the phase of Phi
contour2 = mlab.pipeline.set_active_attribute(contour,
                                    point_scalars='angle')

# # And we display the surface. The colormap is the current attribute: the phase.
mlab.pipeline.surface(contour2, colormap='hsv')

mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)

mlab.show()

import matplotlib.pyplot as plt
import numpy as np

# Assuming Phi is your 3D dataset and already reshaped into (N, N, N)
N = Phi.shape[0]  # Assuming Phi is cubic (N x N x N)
middle_index = N // 2  # Index for the middle slice

# Take a slice along the desired axis, e.g., the z-axis
slice_data = np.abs(Phi[:, :, middle_index])  # Taking the magnitude for visualization

# Plot the slice
# plt.figure(figsize=(8, 6))
# plt.imshow(slice_data, extent=[0, N, 0, N], origin='lower', cmap='viridis')
# plt.colorbar(label='|Phi|')  # Add a colorbar for scale
# plt.title('Middle Slice of the Scalar Field (|Phi|)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()