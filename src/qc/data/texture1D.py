#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Factory for texture generation."""
import cupy as cp


class Texture1D:
    NUMVECTORS = 1
    NUMBITS = 32

    def __init__(self, x_len: int):

        self.x_len = x_len

    def initial_texture(self):
        """initial_texture."""
        x_initial_vector = cp.random.randn(self.x_len,
                                           Texture1D.NUMVECTORS,
                                           dtype=cp.float32)
        #
        # x_reshaped = cp.reshape(x_initial_vector,(self.x_len,self.y_len,self.z_len))

        return self.texture_from_ndarray(x_initial_vector)

    def ones(self):
        """fills texture with ones.
        for checking if derivative calc is correct"""
        x_initial_vector = cp.ones((self.x_len,
                                    Texture1D.NUMVECTORS), dtype=cp.float32)
        # x_reshaped = cp.reshape(x_initial_vector, (self.x_len, self.y_len, self.z_len))
        return self.texture_from_ndarray(x_initial_vector)

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

    import cupy as cp
    import cupy.cuda as cuda

    def create_cuda_texture(array: cp.ndarray):
        """
        Create a CUDA texture from a 1D CuPy array.

        Args:
            array (cp.ndarray): The input CuPy array of shape (N,).

        Returns:
            cuda_texture (int): The handle to the CUDA texture.
        """
        # Ensure the array is contiguous and of the right data type
        if not array.flags.c_contiguous:
            array = cp.ascontiguousarray(array)

        # Define texture reference
        texture_ref = cp.cuda.TextureObject()

        # Create a texture from the CuPy array
        texture_ref = cp.cuda.texture.create(
            array.data.ptr,  # Pointer to the array data
            array.shape,  # Shape of the array
            array.dtype,  # Data type of the array
            cp.cuda.filterMode.POINT,  # Texture filtering mode
            cp.cuda.addressMode.WRAP  # Addressing mode
        )

        return texture_ref

