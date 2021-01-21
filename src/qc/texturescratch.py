#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checking working of Texture and Surface classes"""
import cupy as cp

import qc.surface as S
import qc.texture as T

test_kernel = cp.RawKernel(
    r'''extern "C" __global__ void test(cudaTextureObject_t texture_input,
                                        cudaSurfaceObject_t surface_output, int XLEN, int YLEN, int ZLEN){
    for(unsigned int z = 0; z < ZLEN; z++){
        for(unsigned int y = 0; y < YLEN; y++){
            for(unsigned int x = 0; x < XLEN; x++){
               float value = tex3D<float>(texture_input, (float)x, float(y), float(z));
               printf("%f ", value);
               value *= 2;
               surf3Dwrite<float>(value,surface_output,x*sizeof(float),y,z);
            }
        }
    }
    printf("\n");
                                 }''',
    'test',
    backend='nvcc')
if __name__ == "__main__":
    XLEN = 2
    YLEN = ZLEN = XLEN
    tex_obj = T.Texture(XLEN, YLEN, ZLEN)
    sur_obj = S.Surface(XLEN, YLEN, ZLEN)

    init_tex = tex_obj.initial_texture()

    init_sur = sur_obj.initial_surface()

    test_kernel.compile()

    start_event = cp.cuda.stream.Event()
    stop_event = cp.cuda.stream.Event()
    stream = cp.cuda.stream.Stream()
    with stream:

        for i in range(3):

            start_event.record()
            test_kernel((1, ), (1, ), (
                init_tex,
                init_sur,
                XLEN,
                YLEN,
                ZLEN,
            ))

            init_tex = tex_obj.texture_from_surface(init_sur)

            stop_event.record()
            stop_event.synchronize()
            print(i)
