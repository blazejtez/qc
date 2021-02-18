#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import qc.configparser
import ply.yacc as yacc


class TestParser(unittest.TestCase):

    def test_parse_string(self):
        inp = "$CONTROL OPTIM ENERGY END$ $OPTIM OPTIMBELT = 100 END$ $RASTER DENSITY = 5 END$ $DATA He -1 1 1 1, H 1 1 1 0 END$"  
        settings = configparser.parse_string(inp)
        test_dict = {"CONTROL" : ['OPTIM', 'ENERGY'],"OPTIM":{"OPTIMBELT" : 100.0}, "RASTER" : {"DENSITY":5.0}, "DATA" : [{"He":[-1.0,1.0,1.0,1.0]}, {"H":[1.0,1.0,1.0,0.0]}]}
        self.assertEqual(test_dict, settings)
       
if __name__ == "__main__":
    unittest.main()
