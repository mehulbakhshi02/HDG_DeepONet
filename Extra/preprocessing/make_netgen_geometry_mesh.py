import numpy as np
from ngsolve import *
from netgen.geom2d import SplineGeometry
def WriteWedge(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	angle = 15 #in degrees
	nps = 1
	nps_wedge = 1
	x_min = 0.0
	x_wedge = 0.5
	x_max = 1.5
	y_min = 0.0
	y_max = 1.0
	y_wedge_max = np.tan(angle*np.pi/180.0) * (x_max-x_wedge)
	inlet = np.linspace(y_max, y_min, nps, endpoint=False)
	lower_wall_1 = np.linspace(x_min,x_wedge, nps, endpoint=False)
	lower_wall_2_x = np.linspace(x_wedge,x_max, nps_wedge, endpoint=False)
	lower_wall_2_y = np.linspace(y_min,y_wedge_max, nps_wedge, endpoint=False)
	outlet = np.linspace(y_wedge_max, y_max, nps, endpoint=False)
	upper_wall = np.linspace(x_max, x_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 4*nps+nps_wedge
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 2 (wedge)
	for i in range(0, nps_wedge):
		line = str(count) + '\t\t' + str(lower_wall_2_x[i]) + '\t\t' + str(lower_wall_2_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps_wedge))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')
def WriteWedgeNS(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	angle = 10 #in degrees
	nps = 1
	nps_wedge = 1
	x_min = 0.0
	x_sym = 0.5
	x_wedge = x_sym + 1.5
	x_max = x_wedge + 0.5
	y_min = 0.0
	y_max = 1.0
	y_wedge_max = np.tan(angle*np.pi/180.0) * (x_max-x_wedge)
	inlet = np.linspace(y_max, y_min, nps, endpoint=False)
	lower_wall_1a = np.linspace(x_min,x_sym, nps, endpoint=False)
	lower_wall_1 = np.linspace(x_sym,x_wedge, nps, endpoint=False)
	lower_wall_2_x = np.linspace(x_wedge,x_max, nps_wedge, endpoint=False)
	lower_wall_2_y = np.linspace(y_min,y_wedge_max, nps_wedge, endpoint=False)
	outlet = np.linspace(y_wedge_max, y_max, nps, endpoint=False)
	upper_wall = np.linspace(x_max, x_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 5*nps+nps_wedge
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1a[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall sym
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 2 (wedge)
	for i in range(0, nps_wedge):
		line = str(count) + '\t\t' + str(lower_wall_2_x[i]) + '\t\t' + str(lower_wall_2_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order goem symmetry wall
	for i in range(0, int((nps))):
		bc_val = 3
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps_wedge))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteWedgeChannel(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	angle = 20 #in degrees
	nps = 1
	nps_wedge = 1
	x_min = 0.0
	x_wedge = 0.2
	x_wedge_max = 0.4
	x_max = 1.
	y_min = 0.0
	y_max = 0.2
	y_wedge_max = np.tan(angle*np.pi/180.0) * (x_wedge_max-x_wedge)
	inlet = np.linspace(y_max, y_min, nps, endpoint=False)
	lower_wall_1 = np.linspace(x_min,x_wedge, nps, endpoint=False)
	lower_wall_2_x = np.linspace(x_wedge,x_wedge_max, nps_wedge, endpoint=False)
	lower_wall_2_y = np.linspace(y_min,y_wedge_max, nps_wedge, endpoint=False)
	lower_wall_3_x = np.linspace(x_wedge_max,x_max, nps_wedge, endpoint=False)
	outlet = np.linspace(y_wedge_max, y_max, nps, endpoint=False)
	upper_wall = np.linspace(x_max, x_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 5*nps+nps_wedge
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 2 (wedge)
	for i in range(0, nps_wedge):
		line = str(count) + '\t\t' + str(lower_wall_2_x[i]) + '\t\t' + str(lower_wall_2_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 3 (flat again)
	for i in range(0, nps_wedge):
		line = str(count) + '\t\t' + str(lower_wall_3_x[i]) + '\t\t' + str(y_wedge_max) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps_wedge))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps_wedge))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteWedgeChannel2(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	angle1 = 15 #in degrees
	angle2 = 35 #in degrees
	nps = 1
	nps_wedge = 1
	x_min = 0.0
	x_wedge = 1.
	x_wedge_max_1 = 2.0
	x_wedge_max_2 = 3.0
	a = np.cos(angle1*np.pi/180.0)/np.cos(angle2*np.pi/180.0)
	x_wedge_max_1 = (a*x_wedge_max_2+1.0)/(1.0+a)
	x_max = 3.5
	y_min = 0.0
	y_max = 1.5
	y_wedge_max_1 = np.tan(angle1*np.pi/180.0) * (x_wedge_max_1-x_wedge)
	y_wedge_max_2 = np.tan(angle2*np.pi/180.0) * (x_wedge_max_2-x_wedge_max_1)
	inlet = np.linspace(y_max, y_min, nps, endpoint=False)
	lower_wall_1 = np.linspace(x_min,x_wedge, nps, endpoint=False)
	lower_wall_2_x = np.linspace(x_wedge,x_wedge_max_1, nps_wedge, endpoint=False)
	lower_wall_2_y = np.linspace(y_min,y_wedge_max_1, nps_wedge, endpoint=False)
	lower_wall_3_x = np.linspace(x_wedge_max_1,x_wedge_max_2, nps_wedge, endpoint=False)
	lower_wall_3_y = np.linspace(y_wedge_max_1,y_wedge_max_2+y_wedge_max_1, nps_wedge, endpoint=False)
	lower_wall_4_x = np.linspace(x_wedge_max_2,x_max, nps_wedge, endpoint=False)
	outlet = np.linspace(y_wedge_max_2+y_wedge_max_1, y_max, nps, endpoint=False)
	upper_wall = np.linspace(x_max, x_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 5*nps+2*nps_wedge
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 2 (wedge)
	for i in range(0, nps_wedge):
		line = str(count) + '\t\t' + str(lower_wall_2_x[i]) + '\t\t' + str(lower_wall_2_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 2 (wedge)
	for i in range(0, nps_wedge):
		line = str(count) + '\t\t' + str(lower_wall_3_x[i]) + '\t\t' + str(lower_wall_3_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# # Write the lower wall 3 (flat again)
	for i in range(0, nps_wedge):
		line = str(count) + '\t\t' + str(lower_wall_4_x[i]) + '\t\t' + str(y_wedge_max_2+y_wedge_max_1) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps_wedge))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps_wedge))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# # We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps_wedge))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteExpansion(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	angle = 10 #in degrees
	angle_rad = angle * np.pi/180.
	nps = 1
	side = 1.0
	x_min = 0.0
	x_wedge = side
	x_max = x_wedge + side * np.cos(angle_rad)
	y_max = -side *np.sin(angle_rad)
	x_out = x_wedge + side * np.sin(angle_rad) + side * np.cos(angle_rad)
	y_out = -side * np.sin(angle_rad) + side * np.cos(angle_rad)
	y_top = side
	y_min = 0.0
	inlet = np.linspace(y_top, y_min, nps, endpoint=False)
	lower_wall_1 = np.linspace(x_min,x_wedge, nps, endpoint=False)
	lower_wall_2_x = np.linspace(x_wedge, x_max, nps, endpoint=False)
	lower_wall_2_y = np.linspace(y_min,y_max, nps, endpoint=False)
	outlet_x = np.linspace(x_max, x_out, nps, endpoint=False)
	outlet_y = np.linspace(y_max, y_out, nps, endpoint=False)
	upper_wall_1_x = np.linspace(x_out, x_wedge, nps, endpoint=False)
	upper_wall_1_y = np.linspace(y_out, y_top, nps, endpoint=False)
	upper_wall = np.linspace(x_wedge, x_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 6*nps
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 2 (wedge)
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_2_x[i]) + '\t\t' + str(lower_wall_2_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(outlet_x[i]) + '\t\t' + str(outlet_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 1
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall_1_x[i]) + '\t\t' + str(upper_wall_1_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 2
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(y_top) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteDoubleWedge(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	angle = 10 #in degrees
	angle_rad = angle * np.pi/180.
	nps = 1
	side_x = 1.0
	side_y = 2.0
	x_min = 0.0
	y_min = 0.0
	x_wedge = side_x

	x_max = x_wedge + side_x * np.cos(angle_rad)
	y_max = y_min + side_x * np.sin(angle_rad)

	x_out = x_wedge + side_x * np.cos(angle_rad)
	y_out = side_y - side_x * np.sin(angle_rad)

	y_top = side_y

	inlet = np.linspace(y_top, y_min, nps, endpoint=False)
	lower_wall_1 = np.linspace(x_min,x_wedge, nps, endpoint=False)
	lower_wall_2_x = np.linspace(x_wedge, x_max, nps, endpoint=False)
	lower_wall_2_y = np.linspace(y_min,y_max, nps, endpoint=False)

	outlet = np.linspace(y_max, y_out, nps, endpoint=False)
	upper_wall_1_x = np.linspace(x_out, x_wedge, nps, endpoint=False)
	upper_wall_1_y = np.linspace(y_out, y_top, nps, endpoint=False)
	upper_wall = np.linspace(x_wedge, x_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 6*nps
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 2 (wedge)
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_2_x[i]) + '\t\t' + str(lower_wall_2_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 1
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall_1_x[i]) + '\t\t' + str(upper_wall_1_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 2
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(y_top) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall 2 (wedge)
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteNaca(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1000\n')
	# Number of points in the geometry
	f.write('points\n')
	nps_naca = 100
	nps_out = 4
	r_out = 2000
	theta_out = np.linspace(0, 2.0*np.pi, nps_out, endpoint=False)
	# x_out = r_out * np.cos(theta_out)
	# y_out = r_out * np.sin(theta_out)
	x_out = [-r_out/2, r_out/2, r_out/2, -r_out/2]
	y_out = [-r_out/2, -r_out/2, r_out/2, r_out/2]
	x_mid = 0.0
	y_mid = 0.0
	nps_split_1 = int(nps_naca/2.0)
	nps_split_2 = nps_naca - nps_split_1
	x_mid_sp = 0.3
	x_foil_s1 = np.linspace(1.0,x_mid_sp, nps_split_2, endpoint=False)
	x_foil_s2 = np.linspace(x_mid_sp,0.0, nps_split_1, endpoint=False)
	x_foil_down = np.concatenate([x_foil_s1, x_foil_s2])
	# x_foil_down = np.linspace(1.0,0, nps_naca, endpoint=False)
	y_foil_down = -0.6*(0.2969*np.sqrt(x_foil_down) - 0.1260*x_foil_down - 0.3516*x_foil_down**2.0+0.2843*x_foil_down**3.0-0.1036*x_foil_down**4.0)
	# x_foil_up = np.linspace(0.0,1.0, nps_naca, endpoint=False)
	x_foil_s1 = np.linspace(0.0,x_mid_sp, nps_split_1, endpoint=False)
	x_foil_s2 = np.linspace(x_mid_sp,1.0, nps_split_2, endpoint=False)
	x_foil_up = np.concatenate([x_foil_s1, x_foil_s2])

	y_foil_up = 0.6*(0.2969*np.sqrt(x_foil_up) - 0.1260*x_foil_up - 0.3516*x_foil_up**2.0+0.2843*x_foil_up**3.0-0.1036*x_foil_up**4.0)
	x_foil = np.concatenate([x_foil_down, x_foil_up])
	y_foil = np.concatenate([y_foil_down, y_foil_up])
	for i in range(0, len(x_foil)):
		x_foil[i] = x_foil[i] + x_mid
		y_foil[i] = y_foil[i] + y_mid

	# f.write(str(4*nps+1)+'\n')
	tot_points_in = nps_naca
	tot_points_out = nps_out
	count = 1
	# # Write the aerofoil
	for i in range(0, 2*nps_naca):
		line = str(count) + '\t\t' + str(x_foil[i]) + '\t\t' + str(y_foil[i]) + '\n'
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
	# We want to write the island
	for i in range(0, int(2*(nps_naca))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%(2*nps_naca)+1) + '\t\t' + str((count+1)%(2*nps_naca)+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	count = 0
	# We want to write the outer circle
	for i in range(0, int((nps_out))):
		bc_val = 1
		line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points_out+1+2*nps_naca) + '\t\t' + str((count+1)%tot_points_out+1+2*nps_naca) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteDiamond(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	nps_naca = 4
	nps_out = 4
	r_out = 10
	# theta_out = np.linspace(0, 2.0*np.pi, nps_out, endpoint=False)
	# x_out = r_out * np.cos(theta_out)
	# y_out = r_out * np.sin(theta_out)
	# x_out = [-r_out/4, 3*r_out/4, 3*r_out/4, -r_out/4]
	# y_out = [-r_out/2, -r_out/2, r_out/2, r_out/2]
	x_out = [-r_out/2, r_out/2, r_out/2, -r_out/2]
	y_out = [-r_out/2, -r_out/2, r_out/2, r_out/2]

	# x_out = [-r_out/2, -r_out/4, r_out/4, r_out/2, r_out/2, r_out/2, r_out/2, r_out/4, -r_out/4, -r_out/2, -r_out/2, -r_out/2]
	# y_out = [-r_out/2, -r_out/2, -r_out/2, -r_out/2, -r_out/4, r_out/4, r_out/2, r_out/2, r_out/2, r_out/2, r_out/4, -r_out/4]
	x_mid = 0.0
	y_mid = 0.0
	# angle = 10 # In degrees
	chord = 1.0
	x_foil = [0.5, 0.0, -0.5, 0.0]
	y_max = 0.07/2.0#, np.tan(angle*np.pi/180)*chord/2.0
	y_foil = [0.0, -y_max, 0.0, y_max]
	# for i in range(0, len(x_foil)):
	# 	# x_foil[i] = x_foil[i] + x_mid
	# 	# y_foil[i] = y_foil[i] + y_mid
	# 	print(len(x_foil), x_foil[i])
	# # f.write(str(4*nps+1)+'\n')
	tot_points_in = nps_naca
	tot_points_out = nps_out
	count = 1
	# # Write the aerofoil
	for i in range(0, nps_naca):
		line = str(count) + '\t\t' + str(x_foil[i]) + '\t\t' + str(y_foil[i]) + '\n'
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
	# We want to write the island
	for i in range(0, int((nps_naca))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%(nps_naca)+1) + '\t\t' + str((count+1)%(nps_naca)+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	count = 0
	# # We want to write the outer circle
	# for i in range(0, 1):
	# 	bc_val = 1
	# 	line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points_out+1+nps_naca) + '\t\t' + str((count+1)%tot_points_out+1+nps_naca) + '\t\t'+'-bc='+str(bc_val)+'\n'
	# 	count = count + 1
	# 	f.write(line)

	for i in range(0, int((nps_out))):
		bc_val = 1
		line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points_out+1+nps_naca) + '\t\t' + str((count+1)%tot_points_out+1+nps_naca) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)

	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteDiamond2(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	nps_naca = 3
	nps_out = 4
	r_out = 5
	# theta_out = np.linspace(0, 2.0*np.pi, nps_out, endpoint=False)
	# x_out = r_out * np.cos(theta_out)
	# y_out = r_out * np.sin(theta_out)
	x_out = [-r_out/2, r_out/2, r_out/2, -r_out/2]
	y_out = [-r_out/2, -r_out/2, r_out/2, r_out/2]
	x_mid = 0.0
	y_mid = 0.0
	angle = 10 # In degrees
	chord = 1.0
	# x_foil = [0.5, 0.0, -0.5, 0.0]
	# y_max = np.tan(angle*np.pi/180)*chord/2.0
	# y_foil = [0.0, -y_max, 0.0, y_max]
	height_diamond = np.tan(angle*np.pi/180)*chord/2.0
	dz = height_diamond*3.0
	x_foil_1 = [chord/2.0, 0.0, -chord/2.0]
	y_foil_1 = [2.0*height_diamond+dz, dz, 2.0*height_diamond+dz]

	x_foil_2 = [chord/2.0, -chord/2.0, 0.0]
	y_foil_2 = [-2.0*height_diamond-dz, -2.0*height_diamond-dz, -dz]

	for i in range(0, len(x_foil_2)):
		x_foil_1[i] = x_foil_1[i] + 0.0
		y_foil_1[i] = y_foil_1[i] + height_diamond
		x_foil_2[i] = x_foil_2[i] + 0.0
		y_foil_2[i] = y_foil_2[i] - height_diamond

	# f.write(str(4*nps+1)+'\n')
	tot_points_in = nps_naca
	tot_points_out = nps_out
	count = 1
	# # Write the aerofoil
	for i in range(0, nps_naca):
		line = str(count) + '\t\t' + str(x_foil_1[i]) + '\t\t' + str(y_foil_1[i]) + '\n'
		count = count + 1
		f.write(line)
	for i in range(0, nps_naca):
		line = str(count) + '\t\t' + str(x_foil_2[i]) + '\t\t' + str(y_foil_2[i]) + '\n'
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
	# We want to write the island
	for i in range(0, int((nps_naca))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%(nps_naca)+1) + '\t\t' + str((count+1)%(nps_naca)+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	count = 0
	for i in range(0, int((nps_naca))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%(nps_naca)+1+nps_naca) + '\t\t' + str((count+1)%(nps_naca)+1+nps_naca) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	count = 0
	# We want to write the outer circle
	for i in range(0, int((nps_out))):
		bc_val = 1
		line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points_out+1+2*nps_naca) + '\t\t' + str((count+1)%tot_points_out+1+2*nps_naca) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteScramjet(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	nps = 1
	inlet = np.linspace(3.5, -3.5, nps, endpoint=False)
	lower_wall_x = [0.0, 0.4, 4.9, 12.6, 14.25]
	lower_wall_y = [-3.5, -3.5, -2.9, -2.12, -1.92]
	outlet = np.linspace(-1.7, 1.7, nps, endpoint=False)
	upper_wall_x = [16.9, 14.25, 12.6, 4.9, 0.4]
	upper_wall_y = [1.7, 1.92, 2.12, 2.9, 3.5]

	x_foil_1 = [4.9, 12.6, 14.25, 9.4, 8.9]
	y_foil_1 = [1.4, 1.4, 1.2, 0.5, 0.5]

	x_foil_2 = [4.9, 8.9, 9.4, 14.25, 12.6]
	y_foil_2 = [-1.4, -0.5, -0.5, -1.2, -1.4]

	# f.write(str(4*nps+1)+'\n')
	nps_naca = len(x_foil_1)
	nps_dom = len(upper_wall_y)
	tot_points = 2*nps_dom+2
	count = 1
	# Write the lower wall 
	for i in range(0, nps_dom):
		line = str(count) + '\t\t' + str(lower_wall_x[i]) + '\t\t' + str(lower_wall_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(16.9) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 
	for i in range(0, nps_dom):
		line = str(count) + '\t\t' + str(upper_wall_x[i]) + '\t\t' + str(upper_wall_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(0.0) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)

	for i in range(0, nps_naca):
		line = str(count) + '\t\t' + str(x_foil_1[i]) + '\t\t' + str(y_foil_1[i]) + '\n'
		count = count + 1
		f.write(line)

	for i in range(0, nps_naca):
		line = str(count) + '\t\t' + str(x_foil_2[i]) + '\t\t' + str(y_foil_2[i]) + '\n'
		count = count + 1
		f.write(line)

	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps_dom))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps_dom))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	count = 0
	# We want to write the outer circle
	for i in range(0, int((nps_naca))):
		bc_val = 2
		line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%nps_naca+1+tot_points) + '\t\t' + str((count+1)%nps_naca+1+tot_points) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	count = 0
	for i in range(0, int((nps_naca))):
		bc_val = 2
		line = '1' + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%nps_naca+1+tot_points+nps_naca) + '\t\t' + str((count+1)%nps_naca+1+tot_points+nps_naca) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)

	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteGaussBump(filename):
	# Try and write the geometry file for netgen
	f = open(filename+"_nonlin.in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('\npoints\n')
	order = 3
	nps = 1
	if(order>3):
		print("Order greater than 3 does not work!")
	nps_l = 5*order*(order-1)
	inlet = np.linspace(0.8, 0, nps, endpoint=False)
	# lower_wall_1x = np.linspace(-1.5, -1.17, 1, endpoint=False)
	# lower_wall_1y = 0.0*lower_wall_1x
	# lower_wall_2x = np.linspace(1.17, 1.5, 1, endpoint=False)
	# lower_wall_2y = 0.0*lower_wall_2x
	# lower_wall_bump_x = np.linspace(-1.17,1.17, nps_l-1, endpoint=False)
	lower_wall_bump_x = np.linspace(-1.5,1.5, nps_l, endpoint=False)
	lower_wall_bump_y = 0.0625*np.exp(-25.0*lower_wall_bump_x*lower_wall_bump_x)
	# lower_wall_x = np.concatenate([lower_wall_1x, lower_wall_bump_x, lower_wall_2x])
	# lower_wall_y = np.concatenate([lower_wall_1y, lower_wall_bump_y, lower_wall_2y])
	lower_wall_x = lower_wall_bump_x
	lower_wall_y = lower_wall_bump_y

	upper_wall = np.linspace(1.5, -1.5, nps, endpoint=False)
	outlet = np.linspace(0.0, 0.8, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 3*nps+nps_l
	count = 1
	# Write the lower wall 
	for i in range(0, nps_l):
		line = str(count) + '\t\t' + str(lower_wall_x[i]) + '\t\t' + str(lower_wall_y[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(1.5) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(0.8) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(-1.5) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps_l)/(order-1))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(order) + '\t\t' #+ str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' #+ str((count+2)%tot_points+1) +'\t\t' 
		for p in range(0, order):
			line = line + str((count+p)%tot_points+1) + '\t\t'
		line = line + '-bc='+str(bc_val)+ '\n'
		count = count + order-1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')


	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('\npoints\n')
	nps = 1
	nps_l_lin = int(nps_l/(order-1))
	tot_points = 3*nps+nps_l_lin
	count = 1
	# Write the lower wall 
	for i in range(0, nps_l_lin):
		val = i*(order-1)
		line = str(count) + '\t\t' + str(lower_wall_x[val]) + '\t\t' + str(lower_wall_y[val]) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(1.5) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall[i]) + '\t\t' + str(0.8) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(-1.5) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the lower wall with 1st order geometry
	for i in range(0, int((nps_l_lin))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+ '\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')

def WriteFlatPlate(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')
	nps = 1
	x_min = -1.25
	x_plate = 0.0
	x_max = 1.0
	y_min = 0.0
	y_max = 2.0
	# x_min = -1.0/3.0
	# x_plate = 0.0
	# x_max = 2.0
	# y_min = 0.0
	# y_max = 1.0

	symmetry = np.linspace(x_min,x_plate, nps, endpoint=False)
	wall = np.linspace(x_plate,x_max, nps, endpoint=False)
	outlet = np.linspace(y_min, y_max, nps, endpoint=False)
	farfield = np.linspace(x_max, x_min, nps, endpoint=False)
	inlet = np.linspace(y_max, y_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 5*nps
	count = 1
	# Write the farfield
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(farfield[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
		# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the symmetry
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(symmetry[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the wall
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(wall[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)

	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the far field with 1st order
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet wall with 1st order
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the symmetry with 1st order geometry
	for i in range(0, int((nps))):
		bc_val = 3
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val) +'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the wall
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val) +'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the outlet 
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)

	f.write('\n\nmaterials\n1\t\tdomain1')
def WriteCircle(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('5\n')
	# Number of points in the geometry
	f.write('points\n')
	nps_out = 100

	r_out = 1
	theta_out = np.linspace(0, 2.0*np.pi, nps_out, endpoint=False)
	x_out = r_out * np.cos(theta_out)
	y_out = r_out * np.sin(theta_out)
	tot_points_out = nps_out
	count = 1
	# Write the inlet
	for i in range(0, nps_out):
		line = str(count) + '\t\t' + str(x_out[i]) + '\t\t' + str(y_out[i]) + '\n'
		count = count + 1
		f.write(line)
	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the outer circle
	for i in range(0, int((nps_out/2.))):
		bc_val = 1
		line = '1' + '\t\t' + '0' + '\t\t' + str(3) + '\t\t' + str((count)%tot_points_out+1) + '\t\t' + str((count+1)%tot_points_out+1) + '\t\t'+str((count+2)%tot_points_out+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 2
		f.write(line)
	f.write('\n\nmaterials\n1\t\tdomain1')
	f.close()

def WriteBackwardFacingStep(filename):
	# Try and write the geometry file for netgen
	f = open(filename+".in2d", "w")
	# Key word for the 2d geometry file
	f.write('splinecurves2dv2\n')
	# Grading factor
	f.write('1\n')
	# Number of points in the geometry
	f.write('points\n')

	nps = 1
	x_min = -130
	x_sym = -110
	x_step = 0
	x_max = 50
	y_max = 9
	y_min = 0
	y_step = 1

	inlet = np.linspace(y_max, y_step, nps, endpoint=False)
	symmetry = np.linspace(x_min,x_sym, nps, endpoint=False)
	lower_wall_1 = np.linspace(x_sym,x_step, nps, endpoint=False)
	step = np.linspace(y_step,y_min, nps, endpoint=False)
	lower_wall_2 = np.linspace(x_step, x_max, nps, endpoint=False)
	outlet = np.linspace(y_min, y_max, nps, endpoint=False)
	upper_wall_1 = np.linspace(x_max, x_sym, nps, endpoint=False)
	upper_wall_2 = np.linspace(x_sym, x_min, nps, endpoint=False)
	# f.write(str(4*nps+1)+'\n')
	tot_points = 8*nps
	count = 1
	# Write the lower wall 
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(symmetry[i]) + '\t\t' + str(y_step) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall 1
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_1[i]) + '\t\t' + str(y_step) + '\n'
		count = count + 1
		f.write(line)
	# Write the step
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_step) + '\t\t' + str(step[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the lower wall
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(lower_wall_2[i]) + '\t\t' + str(y_min) + '\n'
		count = count + 1
		f.write(line)
	# Write the outlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_max) + '\t\t' + str(outlet[i]) + '\n'
		count = count + 1
		f.write(line)
	# Write the top wall
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall_1[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(upper_wall_2[i]) + '\t\t' + str(y_max) + '\n'
		count = count + 1
		f.write(line)
	# Write the inlet
	for i in range(0, nps):
		line = str(count) + '\t\t' + str(x_min) + '\t\t' + str(inlet[i]) + '\n'
		count = count + 1
		f.write(line)

	# Number of segments in the geometry
	f.write('\nsegments\n')
	count = 0
	# We want to write the symmetry
	for i in range(0, int((nps))):
		bc_val = 3
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the lower wall
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str(count%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t' + '-bc='+str(bc_val)+'\n'#+ str((count+2)%tot_points+1) +'\t\t' 
		count = count + 1
		f.write(line)
	# We want to write the step
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the lower wall
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the outlet
	for i in range(0, int((nps))):
		bc_val = 5
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall 1
	for i in range(0, int((nps))):
		bc_val = 2
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the upper wall 2
	for i in range(0, int((nps))):
		bc_val = 3
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)
	# We want to write the inlet
	for i in range(0, int((nps))):
		bc_val = 4
		line = str('1') + '\t\t' + '0' + '\t\t' + str(2) + '\t\t' + str((count)%tot_points+1) + '\t\t' + str((count+1)%tot_points+1) + '\t\t'+'-bc='+str(bc_val)+'\n'
		count = count + 1
		f.write(line)

	f.write('\n\nmaterials\n1\t\tdomain1')


name = "doublewedge"
# WriteWedge(name)
WriteWedgeChannel2(name)
# WriteGaussBump(name)
# WriteBackwardFacingStep(name)
geom = SplineGeometry(name+".in2d")
mesh = Mesh( geom.GenerateMesh(maxh=1000000000.))
mesh.ngmesh.Save(name+".vol")