import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

df = pd.read_csv('uplot.csv')

x = df['x']
y = df['y']
f = df['f']

xi = np.linspace(x.min(), x.max(), 50)
yi = np.linspace(y.min(), y.max(), 50)
X, Y = np.meshgrid(xi, yi)

F = griddata((x, y), f, (X, Y), method='cubic')

# 3D Surface Plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, F, cmap='viridis', edgecolor='k')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x, y)')
ax1.set_title('3D Surface Plot')

# Contour Plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, F, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot')

plt.tight_layout()
plt.show()
