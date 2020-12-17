#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import constants
import raster
import hydrogenpsi

def analytical():
    const = constants.Constants()
    radius = 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n",
        "--numerical",
        action = "store_true",
        help="compute hydrogen wavfunctions and energy numericaly")
    group.add_argument(
        "-a",
        "--analytical",
        action = "store_true",
        help="compute hydrogen wavefunctions and energy analyticaly")
    parser.add_argument("principal",
                        type=int,
                        choices=[1, 2, 3, 4],
                        help="principal quantum number",
                        )
    parser.add_argument("folder", help="directory to save the 3D drawings of the modules squared of the wavefunctions")
    args = parser.parse_args()
    if args.numerical:
        print("numerical")
    if args.analytical:
        print("analytical")
    print(args.principal)
    print(args.folder)
