#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Factory for texture generation."""
import cupy as cp


class Texture3D:
    NUMVECTORS = 1
    NUMBITS = 32

    def __init__(self, x_len: int, y_len: int, z_len: int):

        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len

    def initial_texture(self):
        """initial_texture."""
        x_initial_vector = cp.random.randn(self.x_len * self.y_len *
                                           self.z_len,
                                           Texture3D.NUMVECTORS,
                                           dtype=cp.float32)

        x_reshaped = cp.reshape(x_initial_vector,(self.x_len,self.y_len,self.z_len))

        return self.texture_from_ndarray(x_reshaped)

    def ones(self):
        """fills texture with ones.
        for checking if derivative calc is correct"""
        x_initial_vector = cp.ones((self.x_len * self.y_len * self.z_len,
                                    Texture3D.NUMVECTORS), dtype=cp.float32)
        x_reshaped = cp.reshape(x_initial_vector, (self.x_len, self.y_len, self.z_len))
        return self.texture_from_ndarray(x_reshaped)

    def x(self):
        """fills texture with x."""
        x_initial_vector = cp.ones((self.x_len * self.y_len * self.z_len,))
        for i,j,k in zip(range(self.x_len//2), range(self.y_len//2), range(self.z_len//2)):
            x_initial_vector[i][j][k] = max(i, j, k)
            x_initial_vector[-i][-j][-k] = max(i, j, k)
        x_reshaped = cp.reshape(x_initial_vector, (self.x_len, self.y_len, self.z_len))
        print(self.texture_from_ndarray(x_reshaped))
        return self.texture_from_ndarray(x_reshaped)

    def texture_from_surface(self, surface_obj):
        """texture_from_surface.

        :param surface_obj: surface object as written by cuda kernels
        """
        cuda_array = surface_obj.ResDesc.cuArr
        resource_descriptor = cp.cuda.texture.ResourceDescriptor(
            cp.cuda.runtime.cudaResourceTypeArray, cuda_array)

        texture_obj = cp.cuda.texture.TextureObject(resource_descriptor,
                                                    self.texture_descriptor)

        return texture_obj

    def texture_from_ndarray(self, array):
        array = cp.asarray(array, dtype=cp.float32)

        channel_descriptor_input = cp.cuda.texture.ChannelFormatDescriptor(
            Texture3D.NUMBITS, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)

        cuda_array_input = cp.cuda.texture.CUDAarray(channel_descriptor_input,
                                                     self.x_len, self.y_len,
                                                     self.z_len)
        cuda_array_input.copy_from(array)

        resource_descriptor_input = cp.cuda.texture.ResourceDescriptor(
            cp.cuda.runtime.cudaResourceTypeArray, cuda_array_input)

        self.texture_descriptor = cp.cuda.texture.TextureDescriptor(
            [
                cp.cuda.runtime.cudaAddressModeWrap,
                cp.cuda.runtime.cudaAddressModeWrap,
                cp.cuda.runtime.cudaAddressModeWrap
            ], cp.cuda.runtime.cudaFilterModePoint,
            cp.cuda.runtime.cudaReadModeElementType)
        texture_obj = cp.cuda.texture.TextureObject(resource_descriptor_input,
                                                    self.texture_descriptor)

        return texture_obj
