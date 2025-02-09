import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import pandas as pd
import math
def f(x):
    return {
         0: 'w',
         1: 'qx',
         2: 'qy',
         3: 'qz',
    }[x]

def plotter(COMP, D, adj, plot_comp, name, clr, lab):	
	filename = '../../'+name
	df = pd.read_csv (filename, skiprows=0, header=None)
	basic_columns = ['ne', 'ndof', 'h']
	lq_errs = []
	for ll in range(0, COMP):
		for dd in range(0, D+1):
			lq_errs.append(f(dd)+str(ll+1))

	adj_errs = ['Exact', 'Aposteriori']
	if(adj):
		df.columns = basic_columns + lq_errs + adj_errs
	else:
		df.columns = basic_columns + lq_errs #+ adj_errs2

	x = df['ndof']**(1.0/2.)
	if(adj):
		if(plot_comp):
			sol = df['Aposteriori']
			lab = lab + ' Aposteriori'
		else:
			sol = df['Exact']
			lab = lab + ' Exact'
	else:
		plot_col = 'w'+str(plot_comp)

		sol = df[plot_col]

	plt.loglog(x, sol, clr+'--o', label=lab)

	err = abs(sol)
	ne = df['ne']
	ndof = df['ndof']
	ind = 0
	return x


def reference_plot(slope, intercept, clr, p, range):
	dof_range = np.logspace(np.log10(np.min(range**p)), np.log10(np.max(range**p)), num=8)
	logY = -(slope)*np.log10((dof_range)**(1.0/p)) + intercept
	Y = 10**logY
	plt.loglog((dof_range)**(1.0/p), Y, clr+'--*', label='Slope='+str(slope))

COMP = 1
D = 2
adj = 0
plot_comp = 1


filename = 'error_data.csv'
x = plotter(COMP, D, adj, plot_comp, filename, 'r', 'L2')

reference_plot(3., 2, 'k', D, x)

plt.legend()
plt.grid(True,which="both",ls="-")
plt.show()