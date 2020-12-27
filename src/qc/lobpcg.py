#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
from qc.hydrogenh import *
from qc.raster import *
from qc.hydrogenh import *
import copy

class HydrogenLOBPCG:

    def __init__(self, h: HydrogenHamiltonian):

        self.h = h
        pass


    def ReyleighRitz(self, S):
        
        STS = np.transpose(S).dot(S)

        D = np.diag(np.sqrt(np.diag(STS)))

        R = scipy.linalg.cholesky(D.dot(STS).dot(D)) 

        
        invR = scipy.linalg.lapack.dtrtri(R)[0] # returns np.inv(R.T)
        print(invR)
        print(R)
        print(R.T*invR)
        AS = self.h.operate_vec(S)
        STAS = np.transpose(S).dot(AS)
        AUX = invR*D*STASD*np.transpose(invR)

        w, v = scipy.linalg.eigh(AUX)
        print(w)
        print(v)

        C = D.dot(np.transpose(invR)).dot(v)

if __name__ == "__main__":
    HYDROGEN_RADIUS = 30
    intvl = P.closed(-HYDROGEN_RADIUS,HYDROGEN_RADIUS)
    box_ = box.box3D(intvl,intvl,intvl)
    r = Raster(10.)
    xl,yl,zl = r.box_linspaces(box_)
    hh = HydrogenHamiltonian(xl,yl,zl) 
    h = HydrogenLOBPCG(hh)
    X = np.random.randn(len(intvl)**3,4)
    h.ReyleighRitz(X)

