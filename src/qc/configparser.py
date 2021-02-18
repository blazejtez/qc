# -*- coding: utf-8 -*-
import ply.yacc as yacc
from configlexer import tokens, reserved

#CONFIG - razem
def p_dict(t):
    'dict : groups'
    t[0] = dict()
    t[0].update(t[1])

def p_settings_plural(t):
    'groups : groups group'
    t[0] = dict()
    t[0].update(t[1])
    t[0].update(t[2])
def p_settings_singular(t):
    'groups : group'
    t[0] = dict()
    t[0].update(t[1])

#GRUPA CONTROL
def p_control_controlgrp(t):
    'group : DOLLAR CONTROLGRP flags GROUPEND DOLLAR'
    t[0] = {'CONTROL' : t[3]}
    
def p_control_flags(t):
    'flags : flags flag'
    t[0] = [t[1],t[2]]
    
def p_control_flag(t):
    'flags : flag'
    t[0] = t[1]
    
def p_flag(t):
    '''flag : ENERGYFLG
            | OPTIMFLG'''
    t[0] = t[1]
    
#GRUPA OPTIM 
def p_optimgrp(t):
    'group : DOLLAR OPTIMFLG KVPAIR GROUPEND DOLLAR'
    t[0] = {'OPTIM' : t[3]}

def p_kvpair(t):
    'KVPAIR : OPTIMBELTKWDATA EQUALS FLOAT'
    t[0] = {t[1] : t[3]}

#GRUPA RASTER
def p_rastergrp(t):
    'group : DOLLAR RASTERGRP KVPAIR GROUPEND DOLLAR'
    t[0] = {'RASTER' : t[3]}

def p_kvpair_1(t):
    'KVPAIR : DENSITYKWDATA EQUALS FLOAT'
    t[0] = {t[1] : t[3]}

#GRUPA DATA
def p_data_datagroup(t):
    'group : DOLLAR DATAGRP DATALIST GROUPEND DOLLAR'
    t[0] = {'DATA' : t[3]}
    
def p_data_list(t):
    'DATALIST : DATALIST COMMA ATOMDATA'
    t[0] = [t[1],t[3]]
    
def p_data_list_single(t):
    'DATALIST : ATOMDATA'
    t[0] = t[1]

def p_data_atomdata(t):
    'ATOMDATA : ATOMNAME NUMBER NUMBER NUMBER NUMBER'
    t[0] = {t[1]: [t[2], t[3], t[4], t[5]]}

def p_number_s(t):
    'NUMBER : FLOAT_SIGNED'
    t[0] = t[1]

def p_numer_u(t):
    'NUMBER : FLOAT'
    t[0] = t[1]
    



# Error rule for syntax errors
def p_error(p):
     print("Syntax error on line: {0}".format(p.lexer.lineno))


def parse_file_line(path):
    parser = yacc.yacc(debug = 1)
    f = open(path, "r")
    settings = []
    while True:
        try:
            inp = f.readline()
            settings.append(parser.parse(inp))
        except EOFError:
            break
        if not inp: continue
    f.close()
    return settings

def parse_file(path):
    parser = yacc.yacc(debug = 1)
    f = open(path, "r")
    inp = f.read()
    f.close()
    result = parser.parse(inp)
    return result

def parse_string(string):
    parser = yacc.yacc(debug = 1)
    result = parser.parse(string)
    return result

if __name__ == "__main__":

# Build the parser
    parser = yacc.yacc(debug=True)
    while True:
        try:
            s = input('enter input > ')
        except EOFError:
            break
        if not s: continue
        result = parser.parse(s)
        print(result)
