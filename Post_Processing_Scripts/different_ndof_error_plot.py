import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.colors
# import galerkin_projection_error

def reference_plot(slope, intercept, clr, p):
	dof_range = np.logspace(1, 4, num=8)
	logY = -(slope)*np.log10((dof_range)**(1.0/p)) + intercept
	Y = 10**logY
	plt.loglog((dof_range)**(1.0/p), Y, clr+'--*', label='Slope='+str(slope))

def plotter_test(name, clr, lab):
	plt.figure(1)
	filename = name
	ne, ndof,err_proj,err_u,err_dep=np.loadtxt(filename, delimiter=',', unpack=True)
	plt.loglog(ndof, err_u, clr, label=lab)

def plotter_default(name, clr, lab):
	plt.figure(1)
	filename = name
	ne, ndof, sqrt_ndof, err_u, err_sig_x, err_sig_y=np.loadtxt(filename, delimiter=',', unpack=True)
	plt.loglog(ndof, err_u, clr, label=lab)

filename = 'ml_err_data.csv'
plotter_test(filename, 'b-*', 'ML error')
filename = 'error_data.csv'
plotter_default(filename, 'r-*', 'Solver err')
reference_plot(3.0, -1, 'k', 2)

plt.legend()
plt.grid(True,which="both",ls="-")
plt.show()
