#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class Contour:
    """Contour. Class for handling isolines for the volumetric data"""

    def __init__(self, volume: np.ndarray):

        self.volume = volume

    def get_isoline(self, percent_of_volume_covered: float) -> float:
        """get_isoline. Computes a real number isoline_threshold such that
        np.sum(volume[volume>isoline_threshol])/np.sum(volume.flatten()) ~= percent_of_volume_covered

        :param percent_of_volume_covered: total of points over the returned threshold
        :type percent_of_volume_covered: float
        :rtype: float
        """

        #mx = volume.flatten.max()
        #mn = volume.flatten.min()

        volume_sorted_ = np.sort(self.volume.flatten())
        x = np.cumsum(volume_sorted_)

        volume_sorted = np.flip(volume_sorted_)
        y = np.cumsum(volume_sorted)
        volume_cumsum = np.cumsum(volume_sorted)
        volume_cumsum /= volume_cumsum[-1]

        fraction = percent_of_volume_covered / 100.

        insertion_point = np.searchsorted(volume_cumsum, fraction)
        insertion_point_bis = len(volume_cumsum[volume_cumsum<=fraction])
        aux = np.minimum(insertion_point, len(volume_sorted)-1)
        isoline_threshold = .5 * (volume_sorted[aux] +
                                  volume_sorted[aux - 1])

        return isoline_threshold

if __name__ == "__main__":

    x = np.abs(np.random.randn(100, 100, 100))

    c = Contour(x)

    v = c.get_isoline(99)

    print(v)

    print(np.sum(x[x > v]) / np.sum(x.flatten()))
