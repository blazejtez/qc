#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest

import numpy as np
import qc.data.texture3D as T
import qc.data.surface as S
import qc.hamiltonian.potential as P
import cupy as cp
class MainTest(unittest.TestCase):

    def test(self):

        shape=(30,30,30)

        xl = np.linspace(-1, 1,shape[0], dtype=np.float32)
        yl = np.linspace(-1, 1,shape[1],  dtype=np.float32)
        zl = np.linspace(-1, 1,shape[2], dtype=np.float32)
        np.random.seed(0)
        cube = np.random.randn(len(xl),len(yl),len(zl)).astype(np.float32)

        p = P.Potential()

        cube_out_numba = p.operate(cube,xl,yl,zl)

        tex_obj = T.Texture3D(*shape)

        sur_obj = S.Surface(*shape)

        tex = tex_obj.texture_from_ndarray(cube)

        sur = sur_obj.initial_surface()

        sur = p.operate_cupy(tex,sur,xl,yl,zl)
        
        cube_out_cupy = sur_obj.get_data(sur)


        self.assertAlmostEqual(np.sum((cube_out_numba-cp.asnumpy(cube_out_cupy))**2), 0.0)




if __name__ == "__main__":

    unittest.main()
