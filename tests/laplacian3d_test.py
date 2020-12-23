import unittest
import qc
class MainTest(unittest.TestCase):

    def test(self):

        s = qc.laplacian3d.Stencils3D()

        shape=(10,10,10)

        lap = qc.laplacian3d.Laplacian3D(shape,s.stencil2)

        cube = np.random.randn(*shape).astype(np.float32)

        cube_out = lap_matcube(cube)
        
        cube_out_numba = lap.matcube_numba(cube)
        
        self.assertAlmostEqual(np.sum((cube_out-cube_out_numbe)**2), 0.0)

if __name__ == "__main__":

    unittest.main()
