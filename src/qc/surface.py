#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Surface object manipulation class, complementary to texture module'''

import cupy as cp


class Surface:
    NUMBITS = 32

    def __init__(self, x_len: int, y_len: int, z_len: int):

        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len

    def initial_surface(self):
        """initial_surface."""

        y = cp.empty((self.x_len*self.y_len*self.z_len, 1), dtype=cp.float32)

        channel_descriptor = cp.cuda.texture.ChannelFormatDescriptor(
            Surface.NUMBITS, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
        cuda_array = cp.cuda.texture.CUDAarray(channel_descriptor,
                                               self.x_len, self.y_len,
                                               self.z_len)

        y_reshaped = cp.reshape(y, (self.x_len, self.y_len, self.z_len))
        cuda_array.copy_from(y_reshaped)
        resource_descriptor = cp.cuda.texture.ResourceDescriptor(
            cp.cuda.runtime.cudaResourceTypeArray, cuda_array)

        # create surface output object
        surface_obj = cp.cuda.texture.SurfaceObject(resource_descriptor)

        return surface_obj

    def get_data(self, surface_obj):
        """get_data. Returns the array (as a 3D cube of dimensions x_len x y_len x z_len) contained
        in the surface object.

        :param surface_obj: input surface object
        """

        y = cp.zeros((self.x_len* self.y_len* self.z_len, 1), dtype=cp.float32)

        y_reshaped = cp.reshape(y, (self.x_len, self.y_len, self.z_len))


        surface_obj.ResDesc.cuArr.copy_to(y_reshaped)


        return y_reshaped
