#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np
from sympy import Symbol
from sympy.abc import r
from sympy.physics.hydrogen import Psi_nlm, R_nl
from sympy.utilities.lambdify import lambdify


class HydrogenRadial:
    def __init__(self, n: int, l: int):
        """__init__.

        :param n: principal quantum number
        :type n: int
        :param l: orbital quantum number l \in 0,...,n-1
        :type l: int
        """
        self._check(n,l)
        expr = R_nl(n, l, r)
        self.f = lambdify(r, expr, modules='numpy')

    def _check(self, n:int,l:int) -> None:
        if l > n-1:
            raise ValueError("l can't be larger than n-1")
    def eval(self, r: np.ndarray):
        return self.f(r)


class HydrogenPsi:
    def __init__(self, n: int, l: int, m: int):
        """__init__.

        :param n: principal quantum number
        :type n: int
        :param l: orbital quantum number l \in 0,...,n-1
        :type l: int
        :param m: magnetic quantum number m \in -l,...,l
        :type m: int
        """
        self._check(n,l,m)
        r = Symbol("r",real=True,positive=True)
        phi = Symbol("phi",real=True)
        theta = Symbol("theta",real=True)
        expr = Psi_nlm(n,l,m,r,phi,theta,1)
        print(expr)
        self.f = lambdify((r,phi,theta),expr,modules='numpy')

    def _check(self, n:int,l:int,m:int) -> None:
        if l > n-1:
            raise ValueError("l can't be larger than n-1")
        if m < -l or m > l:
            raise ValueError("m can't be less than -l and larger than l")

    def eval(self, r,phi,theta):
        return self.f(r,phi,theta)
if __name__ == "__main__":

    print(1. / (2**.5) * .5 * np.exp(-.5))
    rh = HydrogenRadial(1, 0)
    print(rh.eval(np.asarray([[1., 1.], [1., 1.]])))

    print(2 * np.exp(-1))
    hpsi = HydrogenPsi(2,1,1)
    r = np.random.randn(2,2)**2
    phi = np.random.randn(2,2)
    theta = np.random.randn(2,2)
    print(hpsi.eval(r,phi,theta))
