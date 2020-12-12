#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sqlite3


class Constants:
    """Constants.
    Class for managing constants related to quantum chemical computations."""
    def __init__(self, db='./database/constants.db'):
        # open connection to the database
        conn = sqlite3.connect(db)
        c = conn.cursor()
        # create neutrons and protons masses dictionary
        rows = c.execute('SELECT * FROM MASSES').fetchall()
        self.masses = {}
        for row in rows:
            self.masses[row[0]] = row[1]
        # get nuclei data
        rows = c.execute('SELECT * FROM NUCLEI').fetchall()
        self.atomic_masses = {}
        for row in rows:
            self.atomic_masses[row[0]] = row[1]
        # get van der Wals radi
        

    @property
    def proton_mass(self):
        return self.masses['proton']

    @property
    def neutron_mass(self):
        return self.masses['neutron']

    def Z(self, element):
        """Z.
        Gives the nuclear charge for the element given.
        :param element: element symbol as string 
        """
        return self.atomic_masses[element]


if __name__ == "__main__":
    const = Constants()
    print(const.proton_mass)
    print(const.neutron_mass)
    print(const.Z('H'))
    print(const.Z('He'))
