
import numpy as np 
import csv
import matplotlib.pyplot as plt
import random
import time
from netgen.geom2d import SplineGeometry
from netgen import meshing
from netgen.meshing import Mesh, FaceDescriptor, Element3D,Element2D, Element1D
def ReadPoints(filename):
	x, y=np.loadtxt(filename, delimiter=',', unpack=True)
	x_g = []
	y_g = []
	for i in range(0, len(x)):
		if(i%1==0):
			x_g.append(x[i])
			y_g.append(y[i])
	chord = x_g[0]
	# chord = 1.
	pos = -1
	for i in range(0, len(x_g)):
		x_g[i] = x_g[i]/chord
		if(x_g[i]==0):
			pos = i
	x_fall_down = np.zeros(pos)
	x_fall_up = np.zeros(len(x_g)-pos)
	for i in range(0, pos):
		x_fall_down[i] = x_g[i]
	for i in range(pos, len(x_g)):
		x_fall_up[i-pos] = x_g[i]

	# x_fall_down = x_g[0:pos]
	# print(x_fall_down)
	# print(x_fall_up)
	a0 = 0.594689181
	a1 = 0.298222773
	a2 = -0.127125232
	a3 = -0.357907906
	a4 = 0.291984971
	a5 = -0.105174606
	y_fall_down = -a0*(a1*np.sqrt(x_fall_down) + a2*x_fall_down + a3*x_fall_down**2.0+a4*x_fall_down**3.0+a5*x_fall_down**4.0)
	y_fall_up = a0*(a1*np.sqrt(x_fall_up) + a2*x_fall_up + a3*x_fall_up**2.0+a4*x_fall_up**3.0+a5*x_fall_up**4.0)

	xf = np.concatenate([x_fall_down, x_fall_up])
	yf = np.concatenate([y_fall_down, y_fall_up])
	# print(x_fall_up, y_fall_up)
	# f = open(filename, "r")
	# x = []
	# y = []
	# for line in f:
	# 	val = line.split()
	# 	x.append(float(val[0]))
	# 	y.append(float(val[1]))
	# return x, y
	return xf, yf
def WritePoints(filename, x):
	nps_mokka = 40
	x_mid = 0.0
	y_mid = 0.0
	nps_split_1 = int(nps_mokka/3.0)
	nps_split_2 = nps_mokka - nps_split_1
	x_mid_sp = 0.1
	eps = 0
	x_foil_s1 = np.linspace(1.0,x_mid_sp, nps_split_2, endpoint=False)
	x_foil_s2 = np.linspace(x_mid_sp,eps, nps_split_1, endpoint=False)
	x_fall_down = np.concatenate([x_foil_s1, x_foil_s2])
	# x_fall_down = np.linspace(1.0,0, nps_mokka, endpoint=False)

	# a0 = 0.6
	# a1 = 0.2969
	# a2 = - 0.1260
	# a3 = - 0.3516
	# a4 = 0.2843
	# a5 = - 0.1036
	a0 = 0.594689181
	a1 = 0.298222773
	a2 = - 0.127125232
	a3 = - 0.357907906
	a4 = 0.291984971
	a5 = - 0.105174606
	y_fall_down = -a0*(a1*np.sqrt(x_fall_down) + a2*x_fall_down + a3*x_fall_down**2.0+a4*x_fall_down**3.0+a5*x_fall_down**4.0)
	# x_fall_up = np.linspace(0.0,1.0, nps_mokka, endpoint=False)
	x_foil_s1 = np.linspace(eps,x_mid_sp, nps_split_1, endpoint=False)
	x_foil_s2 = np.linspace(x_mid_sp,1.0, nps_split_2, endpoint=False)
	x_fall_up = np.concatenate([x_foil_s1, x_foil_s2])

	y_fall_up = a0*(a1*np.sqrt(x_fall_up) + a2*x_fall_up + a3*x_fall_up**2.0+a4*x_fall_up**3.0+a5*x_fall_up**4.0)
	x = np.concatenate([x_fall_down, x_fall_up])
	y = np.concatenate([y_fall_down, y_fall_up])
	for i in range(0, len(x)):
		x[i] = x[i] + x_mid
		y[i] = y[i] + y_mid
		p_loc = [x[i], y[i]]
	write_string = filename


	with open(write_string, 'w') as f:
	    writer = csv.writer(f)
	    for i in range(0, len(x)):
	    	writer.writerow((x[i], y[i]))
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

		# if(abs(y1)<1e-10):
		# 	x_mid=(x1)
		# 	y_mid=(m2*x1+c2)
		# elif(abs(y2)<1e-10):
		# 	x_mid=(x2)
		# 	y_mid=(m1*x2+c1)
		# else:
		x_mid=((c1-c2)/(m2-m1))
		y_mid=((m2*c1-m1*c2)/(m2-m1))
		# Hack for cases with inflection point like rae
		if((x_mid>x1 and x_mid>x2) or (x_mid<x1 and x_mid<x2)):
			print("I am here")
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

	# x_out = [-domain,  domain, domain, -domain]
	# y_out = [-domain, -domain, domain, domain]
	# nps_out = len(x_out)
	nps_out = 80
	theta = np.linspace(0, 2*np.pi, nps_out, endpoint=False)
	x_out = domain*np.cos(theta)
	y_out = domain*np.sin(theta)
	nps_mokka = int(len(x_full)/2)


	tot_points_in = nps_mokka
	tot_points_out = nps_out
	

	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	order = 2
	nps_cyl = nps_mokka
	theta_out = np.linspace(0, -2*np.pi, nps_cyl, endpoint=False)
	nps = 1
	tot_points_in = 2*nps_cyl
	tot_points_out = nps_out

	tot_points = nps_out+2*nps_cyl
	count = 1
	count_out = 0
	count_mid = 0
	# # Check for exact coordinates
	# for i in range(0, len(x_full)):
	# 	# if(x_full[i]!=x_full[len(x_full)-i-2]):
	# 	print(x_full[i], x_full[len(x_full)-i-2])
	
	# # Write the aerofoil
	for i in range(0, len(x_full)):
		line = str(count) + '\t\t' + str(round(x_full[i],20)) + '\t\t' + str(round(y_full[i],20)) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps_out):
		line = str(count) + '\t\t' + str(x_out[i]) + '\t\t' + str(y_out[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	for j in range(0, int((len(x_full))/order)):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(order+1) + '\t\t'
		for o in range(0, order+1):
			line = line + str((count+o)%tot_points_in+1) + '\t\t' 
		line = line  + '-bc='+str(bc_val)+'\n'
		count = count + order
		f.write(line)
	count = 0
	# bc_val = 3
	# line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points_out+1+tot_points_in) + '\t\t' + str((count+1)%tot_points_out+1+tot_points_in) + '\t\t'+'-bc='+str(bc_val)+'\n'
	# count = count + 1
	# f.write(line)
	# bc_val = 2
	# line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points_out+1+tot_points_in) + '\t\t' + str((count+1)%tot_points_out+1+tot_points_in) + '\t\t'+'-bc='+str(bc_val)+'\n'
	# count = count + 1
	# f.write(line)
	# We want to write the outer domain
	for i in range(0, int(nps_out)):
		bc_val = 1
		line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points_out+1+tot_points_in) + '\t\t' + str((count+1)%tot_points_out+1+tot_points_in) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)

	f.write('\n\nmaterials\n1\t\tdomain1')
solve = time.time()
name = 'naca_georg.csv'
# name = 'naca_def.csv'
# WritePoints(name)

# Read in the data, synthetic or otherwise
x, y= ReadPoints(name)
# plt.plot(x, y, 'b')
# xc=x[1:]
# yc=y[1:]
xc = []
yc = []
for i in range(0, len(x)):
	xc.append(x[i])
	yc.append(y[i])
xc.append(x[0])
yc.append(y[0])
Ux = ComputeCoefficients(xc)
Uy = ComputeCoefficients(yc)
PlotCurve(Ux, Uy, xc, yc)
x_full, y_full = CalcMid(Ux, Uy, xc, yc)#x_full, y_full = 
# x_mid1 = 0.5*(x[0]+x_full[0])
# y_mid1 = 0.5*(y[0]+y_full[0])
# x_mid_end = 0.5*(x[0]+x_full[-1])
# y_mid_end = 0.5*(y[0]+y_full[-1])
x_fin = []
# x_fin.append(x[0])
# x_fin.append(x_mid1)
y_fin = []
# y_fin.append(y[0])
# y_fin.append(y_mid1)
for i in range(0, len(x_full)-1):
	x_fin.append(x_full[i])
	y_fin.append(y_full[i])
# x_fin.append(x_mid_end)
# y_fin.append(y_mid_end)

geo_name = name[:-4]
WriteGeomFile(x_fin, y_fin, 50, geo_name)
geo = SplineGeometry(geo_name+".in2d")
mesh = geo.GenerateMesh(maxh=10000000000000.4)
mesh.Save(geo_name+".vol")

plt.show()
end = time.time()
print(end-solve)


