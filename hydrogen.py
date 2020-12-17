#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import constants
import raster
import hydrogenpsi
import portion as P
import cart2sph


def analytical(n: int, l: int, m: int):
    """analytical. Computes complex valued Hydrogen wave function for given quantum numbers

    :param n: principal quantum number
    :type n: int
    :param l: orbital quantum number
    :type l: int
    :param m: magnetic quantum number
    :type m: int
    """
    const = constants.Constants()
    radius = const.atomic_radii.au('H')
    r = raster.Raster(250.)
    x_linspace = r.linspace(P.closed(-radius, radius))
    cs = cart2sph.Cart2Sph(x_linspace, x_linspace, x_linspace)
    phi, theta, r = cs.evaluate()
    psi = hydrogenpsi.HydrogenPsi(n, l, m)
    psi_data = psi.eval(r,phi,theta)
    return psi_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n",
        "--numerical",
        action="store_true",
        help="compute hydrogen wavfunctions and energy numericaly")
    group.add_argument(
        "-a",
        "--analytical",
        action="store_true",
        help="compute hydrogen wavefunctions and energy analyticaly")
    parser.add_argument(
        "principal",
        type=int,
        choices=[1, 2, 3, 4],
        help="principal quantum number",
    )
    parser.add_argument(
        "folder",
        help=
        "directory to save the 3D drawings of the modules squared of the wavefunctions"
    )
    args = parser.parse_args()
    if args.numerical:
        print("numerical")
    if args.analytical:
        print("analytical")
    print(args.principal)
    print(args.folder)
    #plot wavefunction computed analytically
    if args.analytical:
        psi = analytical(1,0,0) 

