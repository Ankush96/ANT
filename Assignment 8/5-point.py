import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

ya = xa = 0
yb = xb = 1
dx = dy = 0.05

n = (xb - xa)/dx
n = int(n)
m = (yb - ya)/dy
m = int(m)

x_grid = np.linspace(xa,xb,n+1)
y_grid = np.linspace(ya,yb,m+1)
yv, xv = np.meshgrid(x_grid, y_grid)

Z_n,_ = np.meshgrid(np.zeros(len(y_grid)),np.zeros(len(x_grid)))

max_err = -1e10
prev_max_err = -2e10
allowed_err = 1e-5
flag = True
while flag:
    for i in range(1,n):
        #print '--i = %d--'%i
        x = xa + i*dx
        for j in range(1,m):
            #print 'j = %d'%j
            y = ya + j*dy
            prev = Z_n[i][j]
            Z_n[i][j] = 0.25*(Z_n[i-1][j] + Z_n[i][j-1]
                              + Z_n[i][j+1] + Z_n[i+1][j] 
                              - (x**2 + y**2)/16.0)
            #print abs(Z_n[i][j] - prev)
            if abs(Z_n[i][j] - prev) > max_err:
                max_err = abs(Z_n[i][j] - prev)
    
    if abs(max_err  - prev_max_err) < allowed_err:
        flag = False
    else:
        prev_max_err = max_err

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
wframe = ax.plot_wireframe(xv, yv, Z_n)
plt.show()

