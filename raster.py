#!/usr/bin/env python
# -*- coding: utf-8 -*-

import portion as P
import numpy as np
import math


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
        self._h = 1. / self._density

    @property
    def density(self):

        return self._density

    @property
    def h(self):

        return self._h

    def _interval_len(self, interval: P.interval):

        return interval.upper - interval.lower

    def _round_interval(self, interval: P.interval):
        """_round_interval. Round and center interval

        :param interval: interval in a.u.
        :type interval: P.interval
        """
        # compute rounding
        aux0 = self._interval_len(interval) * self._density
        aux = math.ceil(aux0) * self._h
        aux1 = aux - self._interval_len(interval)
        lower = interval.lower
        upper = interval.upper
        lower -= .5 * aux1
        upper += .5 * aux1
        interval = P.closed(lower, upper)
        return interval

    def linspace(self, interval: P.Interval) -> np.ndarray:
        """linspace.

        :param interval:
        :type interval: P.Interval
        """
        interval = self._round_interval(interval)
        numh = self._interval_len(interval) * self._density
        lin = np.linspace(interval.lower, interval.upper, int(numh) + 1, dtype = np.float32)

        return lin


if __name__ == "__main__":

    r = Raster(15.1)
    i = P.closed(-1.2, 1.2)

    l = r.linspace(i)
    print(l)
