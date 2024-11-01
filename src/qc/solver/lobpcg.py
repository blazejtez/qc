#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import scipy.linalg
import cupy as cp
from qc.data_structure import raster
from qc.hamiltonian.hamiltonian import HamiltonianOperatorCuPy
from qc.data_structure.raster import *


class HydrogenLOBPCG:
    def __init__(self, h: HamiltonianOperatorCuPy):

        self.h = h
        pass

    def RayleighRitz(self, S):

        STS = np.transpose(S).dot(S)
        Ddiag = np.diag(STS)**(-.5)
        D = np.diag(Ddiag)
        Ddiag = Ddiag[..., np.newaxis]
        DSTSD = Ddiag * STS * Ddiag.T

        R = scipy.linalg.cholesky(DSTSD)

        invR = scipy.linalg.lapack.dtrtri(R)[0]  # returns np.inv(R.T)
        AS = self.h.matvec(S).get()
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
        X = cp.array(X, dtype=cp.float32)
        cdiagThetaT = cp.array(diagTheta.T, dtype=cp.float32)
        R = (self.h.matvec(X) - X.dot(cdiagThetaT)).get()
        S = np.empty((m, 2 * nx))
        S[:, :nx] = X.get()
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
    HYDROGEN_RADIUS = 10.
    ALPHA = 0.28294212105225837470023780155114

    interval = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
    box_ = box.box3D(interval, interval, interval)
    r = raster.Raster(10)
    xl, yl, zl = r.box_linspaces(box_)
    N = len(xl) * len(yl) * len(zl)

    hamiltonian = HamiltonianOperatorCuPy(xl, yl, zl, extent=HYDROGEN_RADIUS)

    h = HydrogenLOBPCG(hamiltonian)
    X, Y, Z = cp.meshgrid(cp.array(xl), cp.array(yl), cp.array(zl))
    gaussian_orbital = cp.exp(-ALPHA * (X ** 2 + Y ** 2 + Z ** 2), dtype=cp.float32)
    norm_factor = cp.sum(gaussian_orbital ** 2) * hamiltonian.h ** 3
    gaussian_orbital /= cp.sqrt(norm_factor)
    x0 = cp.reshape(gaussian_orbital, (N, 1)).get()
    h.lobpcgK(x0, 1e-4)
