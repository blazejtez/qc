#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numexpr as ne
import sympy
import mpmath
from numpy import arctan2, sqrt
from sympy import Symbol
from sympy.abc import r
from sympy.physics.hydrogen import Psi_nlm, R_nl, E_nl
from sympy.utilities.lambdify import lambdify


class HydrogenRadial:
    def __init__(self, n: int, l: int):
        """__init__.

        :param n: principal quantum number
        :type n: int
        :param l: orbital quantum number l \in 0,...,n-1
        :type l: int
        """
        self._check(n, l)
        expr = R_nl(n, l, r)
        self.f = lambdify(r, expr, modules='numpy')

    def _check(self, n: int, l: int) -> None:
        if l > n - 1:
            raise ValueError("l can't be larger than n-1")

    def evaluate(self, r: np.ndarray):
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
        self._check(n, l, m)
        r = Symbol("r", real=True, positive=True)
        phi = Symbol("phi", real=True)
        theta = Symbol("theta", real=True)
        x = Symbol("x", real=True)
        y = Symbol("y", real=True)
        z = Symbol("z", real=True)
        expr = sympy.Abs(Psi_nlm(n, l, m, r, phi, theta,
                                 1))**2 * r**2 * sympy.sin(theta)
        expr = expr.subs(r, sympy.sqrt(x**2 + y**2 + z**2)).subs(
            phi, sympy.atan2(y, x)).subs(
                theta, sympy.atan2(
                    sympy.sqrt(x**2 + y**2), z)) * 1 / sympy.sqrt(
                        x**2 + y**2) * 1 / sympy.sqrt(x**2 + y**2 + z**2)
        expr = expr.cancel()

        expr_cmplx = Psi_nlm(n, l, m, r, phi, theta,
                             1) * (r**2 * sympy.sin(theta))**(1./2)
        expr_cmplx = expr_cmplx.subs(r, sympy.sqrt(x**2 + y**2 + z**2)).subs(
            phi, sympy.atan2(y, x)).subs(
                theta, sympy.atan2(sympy.sqrt(x**2 + y**2), z)) * (
                    1 / sympy.sqrt(x**2 + y**2) * 1 /
                    sympy.sqrt(x**2 + y**2 + z**2))**(1./2)
        expr_cmplx = expr_cmplx.cancel()
        self.f = lambdify((x, y, z), expr, modules='numpy')
        self.f_cmplx = lambdify((x, y, z), expr_cmplx, modules='numpy')

    def _check(self, n: int, l: int, m: int) -> None:
        if l > n - 1:
            raise ValueError("l can't be larger than n-1")
        if m < -l or m > l:
            raise ValueError("m can't be less than -l and larger than l")

    def evaluate(self, x, y, z):
        """evaluate.

        :param x: meshgrid for x (as returned by numpy.mgrid)
        :param y: meshgrid for y
        :param z: meshgrid for z
        """
        return self.f(x, y, z)

    def evaluate_complex(self, x, y, z):
        """evaluate_complex.

        :param x: meshgrid for x (as returned by numpy.mgrid)
        :param y: meshgrid for y
        :param z: meshgrid for z
        """
        psi = self.f_cmplx(x, y, z)
        return np.real(psi), np.imag(psi)


class HydrogenEnergy:
    def __init__(self, n: int):
        """__init__.

        :param n: principal quantum number
        :type n: int
        """
        self.E = E_nl(n)

    def eval(self):

        return self.E


if __name__ == "__main__":

    rh = HydrogenRadial(1, 0)

    hpsi = HydrogenPsi(2, 1, 1)
    x = np.random.randn(2, 2)
    y = np.random.randn(2, 2)
    z = np.random.randn(2, 2)
    print(hpsi.evaluate(x, y, z))
    re, im = hpsi.evaluate_complex(x,y,z)
    print(re)
    print(im)
    print(re**2+im**2)
    he = HydrogenEnergy(2)
    print(he.eval())
