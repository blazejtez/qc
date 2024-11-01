import numpy as np
from qc.data_structure.util_cub import load_basic
from mayavi import mlab

Phi = load_basic("h1xx11.cub").get()
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