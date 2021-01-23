import unittest

import numpy as np
import qc.texture as T
import qc.surface as T
from qc.laplacian3d import *


class MainTest(unittest.TestCase):

    def test(self):

        s = Stencils3D()

        shape=(20,20,20)

        lap = Laplacian3D(1.)

        np.random.seed(0)
        cube = np.random.randn(*shape).astype(np.float32)

        #  cube = np.transpose(cube)
         
        #  for x in range(shape[0]):
            #  for y in range(shape[1]):
                #  for z in range(shape[2]):
                    #  print(f"{x}, {y}, {z} = {cube[x,y,z]} ")
        cube_out = lap.matcube(cube, s.stencil7pts)

        cube_out_numba = lap.matcube_numba(cube)
        
        tex_obj = T.Texture(*shape)

        sur_obj = S.Surface(*shape)

        tex = tex_obj.texture_from_ndarray(cube)

        sur = sur_obj.initial_surface()

        sur = lap.matcube_cupy(sur, tex)
        
        cube_out_cupy = sur_obj.get_data(sur)

        self.assertAlmostEqual(np.sum((cube_out-cube_out_numba)**2), 0.0)

        self.assertAlmostEqual(np.sum((cube_out-cp.asnumpy(cube_out_cupy))**2), 0.0)




if __name__ == "__main__":

    unittest.main()
