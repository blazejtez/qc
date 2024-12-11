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

# wavefunction_1s = hydrogen_1s(N, Z, a0, N_1s)
# print(wavefunction_1s.shape)  # Should be (N^3,)




# Phi = load_basic("h100.cub").get()

Phi = hydrogen_2s(N, Z, a0, N_1s).get()
normalizing_factor = Phi.T.dot(Phi)
Phi /= normalizing_factor
Phi = Phi.reshape((N, N, N))
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

# Now we select the 'angle' attribute, ie the phase of Phi
contour2 = mlab.pipeline.set_active_attribute(contour,
                                    point_scalars='angle')

# And we display the surface. The colormap is the current attribute: the phase.
mlab.pipeline.surface(contour2, colormap='hsv')

mlab.colorbar(title='Phase', orientation='vertical', nb_labels=3)

mlab.show()