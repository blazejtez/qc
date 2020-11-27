#!/usr/bin/env python
# -*- coding: utf-8 -*-

import portion
from raster import Raster
import Praktyki.cut_box as box


class Box3DPhysical(box.box3D):
    """Box3DPhysical.
    Class containing the box-like support for the particle wave-function. The boundaries of the box are given in atomic units of length
    that is in units of Bohr radius.
    """
    def __init__(self, interval_x: portion.Interval,
                 interval_y: portion.Interval, interval_z: portion.Interval):
        """__init__.
        Method initializing the Box3DPhysical object from three, one per axis, intervals, which are objects of type portion.Interval.

        :param interval_x: x-axis 
        :type interval_x: portion.Interval
        :param interval_y: y-axis
        :type interval_y: portion.Interval
        :param interval_z: z-axis
        :type interval_z: portion.Interval
        """
        super().__init__(interval_x, interval_y, interval_z)


class Box3DOnGrid(box.box3D):
    """Box3DOnGrid.
    Class containing the box-like support for the particle wav-function. The boundaries of the box are given in terms of number
    of grid cells. The number of grid cells depends on the Raster.density.
    """
    def __init__(self, box: Box3DPhysical, raster: Raster):

        x = box.get_interval_x()
        y = box.get_interval_y()
        z = box.get_interval_z()

        xlr = round(x.lower * raster.density)
        xur = round(x.upper * raster.density)
        ylr = round(y.lower * raster.density)
        yur = round(y.upper * raster.density)
        zlr = round(z.lower * raster.density)
        zur = round(z.upper * raster.density)

        super().__init__(portion.closed(xlr, xur), portion.closed(ylr, yur),
                         portion.closed(zlr, zur))


if __name__ == '__main__':

    b = Box3DPhysical(portion.closed(1.2, 3.), portion.closed(3.2, 4.1),
                      portion.closed(0.5, 0.8))
    print(b)

    rast = Raster(50)


    bog = Box3DOnGrid(b,rast)

    print(bog)
