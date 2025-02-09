import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import pandas as pd
import csv


def plotter_log(D, name, clr, lab):
	filename = '../../'+name
	ne, ndof, sqrt_ndof, time, drag=np.loadtxt(filename, delimiter=',', unpack=True)
	sol = abs(drag)
	h = ndof**(1./D)
	plt.semilogx(h, sol, clr+'--o', label=lab)

D = 2
name = 'log-2-Ma0.15-Al10-Re6e+06-Evr3.csv'
plotter_log(D, name, 'r', 'p2')
plt.legend()
plt.xlabel('h')
plt.ylabel('Drag')
plt.grid(True,which="both",ls="-")
plt.show()
