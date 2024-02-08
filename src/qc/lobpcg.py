#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
from src.qc.hydrogenh import *
from src.qc.raster import *
import copy


class HydrogenLOBPCG:
    def __init__(self, h: HydrogenHamiltonian):

        self.h = h
        pass

    def RayleighRitz(self, S):

        STS = np.transpose(S).dot(S)
        Ddiag = np.diag(STS)**(-.5)
        D = np.diag(Ddiag)
        Ddiag = Ddiag[..., np.newaxis]
        print(Ddiag.shape)
        DSTSD = Ddiag * STS * Ddiag.T

        R = scipy.linalg.cholesky(DSTSD)

        invR = scipy.linalg.lapack.dtrtri(R)[0]  # returns np.inv(R.T)
        AS = self.h.operate_vec(S)
        STAS = np.transpose(S).dot(AS)
        AUX = invR.dot(Ddiag * STAS * Ddiag.T).dot(np.transpose(invR))
        w, v = scipy.linalg.eigh(AUX)
        C = (Ddiag * np.transpose(invR)).dot(v)
        return C, np.diag(w)

    def lobpcgK(self, X: np.ndarray, tau: float):
        C, Theta = self.RayleighRitz(X)
        diagTheta = np.diag(Theta)
        diagTheta = diagTheta[..., np.newaxis]
        nx = np.size(X, 1)
        m = np.size(X, 0)
        X = X.dot(C)
        R = self.h.operate_vec(X) - X * diagTheta.T
        S = np.empty((m, 2 * nx))
        S[:, :nx] = X
        S[:, nx:] = R
        for i in range(20):
            tic = time.time()
            [C, Theta] = self.RayleighRitz(S)
            X = S.dot(C[:, :nx])
            diagTheta = np.diag(Theta)
            print(f'eigenvalue: {np.min(diagTheta)}')
            diagTheta = diagTheta[..., np.newaxis]
            tic1 = time.time()
            R = self.h.operate_vec(X) - X * diagTheta.T[:, :nx]
            toc1 = time.time()
            print(f'time elapsed for R: {toc1-tic1} sec.')
            P = S[:, nx:].dot(C[nx:, :nx])
            S = np.empty((m, 2 * nx + np.size(P, 1)))
            S[:, :nx] = X
            S[:, nx:2 * nx] = R
            S[:, 2 * nx:] = P
            toc = time.time()
            print(f'time elapsed for one iteration: {toc-tic} sec.')

if __name__ == "__main__":
    HYDROGEN_RADIUS = 30
    intvl = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
    box_ = box.box3D(intvl, intvl, intvl)
    r = Raster(10.)
    xl, yl, zl = r.box_linspaces(box_)
    hh = HydrogenHamiltonian(xl, yl, zl)
    h = HydrogenLOBPCG(hh)
    X = np.random.randn(len(xl)**3, 1)
    h.lobpcgK(X, 1e-4)
