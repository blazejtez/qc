# -*- coding: utf-8 -*-

import ply.lex as lex

tokens = [
    'FLOAT_SIGNED',
    'FLOAT',
    'GROUPEND',
    'CONTROLGRP',
    'ENERGYFLG',
    'OPTIMFLG',
    'OPTIMBELTKWDATA',
    'RASTERGRP',
    'DENSITYKWDATA',
    'DATAGRP',
    'DOLLAR',
    'EQUALS',
    'ATOMNAME',
    'COMMA',
]

reserved = {
    'CONTROL': 'CONTROLGRP',
    #control:
    'ENERGY': 'ENERGYFLG',
    'OPTIM': 'OPTIMFLG',
    #opti:
    'OPTIMBELT': 'OPTIMBELTKWDATA', #liczba float wieksza od zera
    'RASTER': 'RASTERGRP',
    #density
    'DENSITY': 'DENSITYKWDATA', #liczba float wieksza od zera
    'DATA': 'DATAGRP',
    'END' : 'GROUPEND',
    
}

atoms = ['H', 'He', 'C']

t_COMMA = r'[,]'
t_EQUALS  = r'='
t_DOLLAR = r'\$'

def t_FLOAT_SIGNED(t):
    r'[-+][0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?'
    t.value = float(t.value)
    return t

def t_FLOAT(t):
    r'[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?'
    t.value = float(t.value)
    return t

def t_ID(t):
    r'[-+]?[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in reserved:
        t.type = reserved[t.value]
    elif t.value in atoms:
        t.type = 'ATOMNAME'
    return t

# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'

# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
lexer = lex.lex(debug = 1)

#lexer.input(data)

#while True:
#    tok = lexer.token()
#    if not tok:
#        break
#    print(tok)
