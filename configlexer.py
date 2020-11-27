#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ply.lex as lex

# list of token names

tokens = [
    'FLOAT',
]

reserved = {
    'raster': 'RASTERGRP',
    'data': 'DATAGRP',
    'density': 'DENSITYKWDATA',
    'end' : 'GROUPEND'
}

literals = ['$', '=']

tokens = tokens + list(reserved.values())

print(tokens)


def t_FLOAT(t):
    r'[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?'
    t.value = float(t.value)
    return t


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in reserved:
        t.type = reserved[t.value]
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


# Build the lexer
lexer = lex.lex()

data = '''$raster density = 30.1 end$'''

lexer.input(data)

while True:
    tok = lexer.token()
    if not tok:
        break
    print(tok)
