#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Raster:
    """Raster.
    Class containing the grid density, i.e. the number of grid cells per atomic units unit length (Bohr radius)
    """

    def __init__(self, density: float):
        """__init__.
        Initialize Raster object with grid density.        

        :param density: number of grid points per unit lenght (Bohr radius), may be fractional
        :type density: float
        """

        self._density = density

    @property
    def density(self):

        return self._density
    
