import numpy as np 
import csv
import matplotlib.pyplot as plt
import random
import time
from netgen.geom2d import SplineGeometry
from netgen import meshing
from netgen.meshing import Mesh, FaceDescriptor, Element3D,Element2D, Element1D

def GenerateQuarterCircle(filename, r):
	nps = 40
	theta = np.linspace(np.pi, np.pi/2., nps)
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	return x, y
def ComputeCoefficients(y):

	n_coeff = 3
	m = len(y)
	f = np.zeros(m)
	A = np.zeros(m*m)

	A[0] = 2.
	A[1] = 1.
	A[m*(m-1)+(m-2)] = 1
	A[m*(m-1)+(m-1)] = 2
	f[0] = 3.*(y[1]-y[0])
	f[len(y)-1] = 3.*(y[-1]-y[-2])
	for i in range(1, m-1):
		f[i] = 3.*(y[i+1]-y[i-1])
		A[i*m+i] = 4
		A[i*m+i-1] = 1
		A[i*m+i+1] = 1
	# return U
	A = np.reshape(A, (m, m))

	# print(f)
	Ainv = np.linalg.inv(A) 
	D = Ainv.dot(f)

	U = np.zeros(4*(m-1))
	for i in range(0, m-1):
		a = y[i]
		b = D[i]
		c = 3.*(y[i+1]-y[i])-2.*D[i]-D[i+1]
		d = 2.*(y[i]-y[i+1])+D[i]+D[i+1]

		ap = i*4+0
		bp = i*4+1
		cp = i*4+2
		dp = i*4+3

		U[ap] = a
		U[bp] = b
		U[cp] = c
		U[dp] = d
	return U

def PlotCurve(Ux,Uy, x, y):

	plt.plot(x, y, 'r*')
	# plt.plot(x, 'bo')
	N = 50
	tg = np.linspace(0, 1, N)
	x_c = []
	y_c = []
	# y_c.append(0)
	# x_c.append(0)
	for i in range(0, len(x)-1):
		ax = Ux[i*4+0]
		bx = Ux[i*4+1]
		cx = Ux[i*4+2]
		dx = Ux[i*4+3]
		ay = Uy[i*4+0]
		by = Uy[i*4+1]
		cy = Uy[i*4+2]
		dy = Uy[i*4+3]
		for j in range(0, N):
			t = tg[j]
			x_loc = ax+bx*t+cx*t*t+dx*t*t*t
			y_loc = ay+by*t+cy*t*t+dy*t*t*t
			x_c.append(x_loc)
			y_c.append(y_loc)

	plt.plot(x_c, y_c, 'r-')
def CalcMid(Ux, Uy, x, y):
	m = len(x)
	x_m = []
	y_m = []
	x_full = []
	y_full = []
	for i in range(0, m-1):
		ax = Ux[i*4+0]
		bx = Ux[i*4+1]
		cx = Ux[i*4+2]
		dx = Ux[i*4+3]
		ay = Uy[i*4+0]
		by = Uy[i*4+1]
		cy = Uy[i*4+2]
		dy = Uy[i*4+3]
		dxdt1 = bx
		dydt1 = by
		# dxdt1 = max(abs(dxdt1), 1e-15)
		dydx1 = dydt1/dxdt1

		dxdt2 = bx+2.*cx+3.*dx
		dydt2 = by+2.*cy+3.*dy
		# dxdt2 = max(dxdt2, 1e-15)
		dydx2 = dydt2/dxdt2

		m1 = dydx1
		m2 = dydx2
		x1 = x[i]
		y1 = y[i]
		x2 = x[i+1]
		y2 = y[i+1]
		c1 = y1-m1*x1
		c2 = y2-m2*x2
		if(abs(y1)<1e-8):
			x_mid=(x1)
			y_mid=(m2*x1+c2)
		elif(abs(y2)<1e-8):
			x_mid=(x2)
			y_mid=(m1*x2+c1)
		else:
			x_mid=((c1-c2)/(m2-m1))
			y_mid=((m2*c1-m1*c2)/(m2-m1))
		# Hack for cases with inflection point like rae
		if((x_mid>x1 and x_mid>x2) or (x_mid<x1 and x_mid<x2)):
			x_mid = 0.5*(x1+x2)
			y_mid = 0.5*(y1+y2)

		x_m.append(x_mid)
		y_m.append(y_mid)
		x_full.append(x1)
		x_full.append(x_mid)
		y_full.append(y1)
		y_full.append(y_mid)

		# if((y_mid>y1 and y_mid>y2) or (y_mid<y1 and y_mid<y2)):
		# 	print(y1, y2, y_mid)
	x_full.append(x[-1])
	y_full.append(y[-1])
	plt.plot(x_full, y_full, 'o')
	plt.plot(x_m, y_m, '*')
	return x_full, y_full
def WriteGeomFile(x_full, y_full, domain, filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	nps = 1
	x_min = 0.0
	x_max = domain
	y_min = 0.0
	y_max = domain
	r_cyl = max(abs(y_full))
	w1 = np.linspace(x_min, x_max-r_cyl, nps, endpoint=False)
	r_out = len(x_full)
	x_full = x_full+x_max

	# w2 = np.linspace(0.5*(x_min+x_max)+r_cyl, x_max, nps, endpoint=False)
	w3 = np.linspace(y_min+r_cyl, y_max, nps, endpoint=False)
	w4 = np.linspace(x_max, x_min, nps, endpoint=False)
	w5 = np.linspace(y_max, y_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 4*nps+r_out-1
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(w1[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
		# Curved point
	for i in range(0, r_out):
		line = str(count) + '\t\t' + str(x_full[i]) + '\t\t' + str(y_full[i]) + '\n'
		count = count + 1
		f.write(line)

	# for i in range(0, nps):
	# 	line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(w3[i]) + '\n'
	# 	count = count + 1
	# 	f.write(line)
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(w4[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(w5[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	order = 2
	count = 0
	for i in range(0, int(nps)):
		bc_val = 3
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' +
		count = count + 1
		f.write(line)
	for j in range(0, int(r_out/order)):
		# # We want to write the lower wall with 1st order geometry
		for i in range(0, int((1))):
			bc_val = 2
			line = str('1') + '\t\t' + '0' + '\t\t' + str(order+1) + '\t\t'
			for o in range(0, order+1):
				line = line + str((count+o)%tot_points+1) + '\t\t' 
			line = line  + '-bc='+str(bc_val)+'\n'
			count = count + order
			f.write(line)
	for i in range(0, int(nps)):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' +
		count = count + 1
		f.write(line)
	for i in range(0, int(nps)):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' +
		count = count + 1
		f.write(line)
	for i in range(0, int(nps)):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' +
		count = count + 1
		f.write(line)


	f.write('\n\nmaterials\n1\t\tdomain1')
	
solve = time.time()

name = 'quarter_circle'
r = 0.5
# Make points
x, y= GenerateQuarterCircle(name, r)
xc=x
yc=y
# Compute piecewise polynomial coefficnets for x and y
Ux = ComputeCoefficients(xc)
Uy = ComputeCoefficients(yc)
# Plot the piecewise curve
PlotCurve(Ux, Uy, xc, yc)
# Calculate midpoints for netgen geometryfile
x_full, y_full = CalcMid(Ux, Uy, xc, yc)
x_full = np.array(x_full)
y_full = np.array(y_full)
geo_name = name
# The third parameter is the domain size around the geometry
WriteGeomFile(x_full, y_full, 10, geo_name)
geo = SplineGeometry(geo_name+".in2d")
mesh = geo.GenerateMesh(maxh=10000000000000.4)
mesh.Save(geo_name+".vol")

# plt.show()
end = time.time()
print(end-solve)


