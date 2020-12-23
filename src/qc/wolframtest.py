#!/usr/bin/env python
# -*- coding: utf-8 -*-


from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import numpy as np
x = np.random.randn(1000,1000,1000)
with WolframLanguageSession('/usr/local/bin/WolframKernel') as session:
    session.start()
    print(session.started)
    print(session.evaluate(wl.Print('Hello world!!!')))
    session.evaluate(wl.Export('plot.pdf',wl.ListContourPlot3D(x, Contours = [0.1]))) 
#session.terminate() 
