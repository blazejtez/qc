#!/usr/bin/env python
# -*- coding: utf-8 -*-

import qc.raster as raster
import cupy as cp
import numpy as np
import qc.hydrogenpsi as hydrogenpsi
import Praktyki.cut_box as box
import portion as P
import qc.hydrogenh as H
import qc.texture as T
import qc.surface as S
if __name__ == "__main__":

    HYDROGEN_RADIUS = 35.
    RASTER_DENSITY = 5
    quantum_numbers = (2,0,0)
    r = raster.Raster(RASTER_DENSITY)

    intvl = P.closed(-HYDROGEN_RADIUS,HYDROGEN_RADIUS)
    box_ = box.box3D(intvl,intvl,intvl)

    xl, yl, zl = r.box_linspaces(box_)
    
    x, y, z = np.meshgrid(xl, yl, zl)
    
    psi = hydrogenpsi.HydrogenPsi(*quantum_numbers)    

    psi_re, psi_im = psi.evaluate_complex(x,y,z)

    psi_laplacian_re, psi_laplacian_im = psi.evaluate_laplacian_complex(x,y,z)
   #  psi_potential_re, psi_potential_im = psi.evaluate_potential_complex(x,y,z)

    #  h_psi_cmplx = psi_laplacian_re + psi_potential_re + 1j*(psi_laplacian_im+psi_potential_im)

    h_psi_cmplx = psi_laplacian_re  + 1j*psi_laplacian_im

    psi_cmplx = psi_re + 1j*psi_im

    psi_re = cp.asarray(psi_re,dtype=cp.float32)
    psi_im = cp.asarray(psi_im,dtype=cp.float32)

    h = H.HydrogenHamiltonian(xl,yl,zl)

    sur = S.Surface(len(xl),len(yl),len(zl))
    tex = T.Texture(len(xl),len(yl),len(zl))

    tex_obj = tex.texture_from_ndarray(psi_re)
    sur_obj = sur.initial_surface()
    Q1 = Q2 = 1.
    sur_out = h.operate_unperturbed_cupy(tex_obj,sur_obj,Q1,Q2)
    h_psi_re = sur.get_data(sur_out)

    tex_obj = tex.texture_from_ndarray(psi_im)
    sur_obj = sur.initial_surface()
    Q1 = Q2 = 1.
    sur_out = h.operate_unperturbed_cupy(tex_obj,sur_obj,Q1,Q2)
    h_psi_im = sur.get_data(sur_out)

    E = cp.sum((psi_re+1j*psi_im)*(h_psi_re-1j*h_psi_im))*(xl[1]-xl[0])**3

    E_sympy = np.sum(psi_cmplx*np.conj(h_psi_cmplx))*(xl[1]-xl[0])**3 

    print(E)

    print(E_sympy)

    he = hydrogenpsi.HydrogenEnergy(quantum_numbers[0])

    print(he.eval())
