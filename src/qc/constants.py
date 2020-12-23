#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sqlite3


class Constants:
    """Constants.
    Class for managing constants related to quantum chemical computations."""
    def __init__(self, db='./database/constants.db'):
        # save db path as an object attribute for latter use
        self.db = db
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
        # create AtomicRadii class
        self.atomic_radii = self._create_atomic_radii()
        # create BohrRadius class
        self.bohr_radius = self._create_bohr_radius()

    class AtomicRadii:
        """AtomicRadii. Class for handling the van der Walls radii"""
        def __init__(self, constants_instance):
            # save outer class instance for latter use
            self.constants = constants_instance
            # open connection to the database
            conn = sqlite3.connect(self.constants.db)
            c = conn.cursor()
            # create atomic radii dictionary
            rows = c.execute('SELECT * FROM "ATOMIC RADII"').fetchall()
            self.d = {}
            for row in rows:
                self.d[row[0]] = row[1]

        def angstrom(self, element):
            """angstrom. Returns the atomic radius expressed in Angstroems

            :param element: symbol of the element
            """
            return self.d[element]
        def au(self, element):
            """au. Returns the atomic radius expressed in atomic units

            :param element: symbol of the element
            """
            return self.d[element]/self.constants.bohr_radius.angstroms

    class BohrRadius:
        """BohrRadius. Class for providing values of Bohr radius in Angstoems"""


        def __init__(self, constants_instance):
            self.constants = constants_instance

            conn = sqlite3.connect(self.constants.db)
            c = conn.cursor()

            rows = c.execute('SELECT * FROM "BOHR RADIUS"').fetchall()
            
            self.bohr = rows[0][0]

        @property
        def angstroms(self):
            return self.bohr

        @property
        def au(self):
            return 1.

    def _create_atomic_radii(self):
        return Constants.AtomicRadii(self)

    def _create_bohr_radius(self):
        return Constants.BohrRadius(self)
    
    @property
    def proton_mass(self):
        """proton_mass."""
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
    print(const.bohr_radius.angstroms) 
    print(const.atomic_radii.au('H'))
