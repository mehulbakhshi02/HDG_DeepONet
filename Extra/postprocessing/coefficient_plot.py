import numpy as np
import re
import matplotlib.pyplot as plt
import csv
import sys

def SortForPlot(x_plot, sol_plot):
	yx = zip(x_plot, sol_plot)
	yx.sort()
	x_s, v_s = zip(*yx)
	return x_s, v_s

def Plotter_coefficient(name, clr, lab):
	filename = '../../'+name
	f = open(filename, "r")
	bcnr = []
	x = []
	y = []
	coeff = []
	cf = []
	sol = []
	for line in f:
		var = line.split()
		bcnr.append(int(var[0]))
		x.append((float(var[1])))
		y.append((float(var[2])))
		coeff.append((float(var[3])))
		cf.append((float(var[4])))
		# sol.append((float(var[3])))
		sol.append(-1*(float(var[4])))


	x_plot = []
	sol_plot = []
	for i in range(0, len(bcnr)):
		if(bcnr[i]==1):
			x_plot.append(x[i])
			sol_plot.append(sol[i])
	x, v = SortForPlot(x_plot, sol_plot)
	
	plt.semilogx(x, v, clr+'--', label=lab)
	

name = 'coefficient.csv'
Plotter_coefficient(name, 'r', lab)
plt.legend()
plt.show()