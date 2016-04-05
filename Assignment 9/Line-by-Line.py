
# coding: utf-8

# In[20]:

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


# In[21]:

def thomas_algo(a, b, c, d):
    n = len(d)
    c_ = [0 for i in range(n)]
    d_ = [0 for i in range(n)]
    y = [0 for i in range(n)]

    c_[0] = c[0] / (1.0 * b[0])
    d_[0] = d[0] / (1.0 * b[0])
    
    for i in range(1,n):
        c_[i] = c[i]/(b[i] - a[i]*c_[i-1])
        d_[i] = (d[i] - a[i]*d_[i-1])/(b[i] - a[i]*c_[i-1])
    
    y[n-1] = d_[n-1]
    for i in range(n-2, -1, -1):
        y[i] = d_[i] - c_[i]*y[i+1]

    return y


# In[22]:

dx = dy = 0.05
xa = ya = 0
xb = yb = 1

m = n = int(1.0/dx)
x_grid = np.linspace(xa,xb,n+1)
y_grid = np.linspace(ya,yb,m+1)
yv, xv = np.meshgrid(x_grid, y_grid)

Z_n,_ = np.meshgrid(np.zeros(len(y_grid)),np.zeros(len(x_grid)))


# In[23]:

flag = True
prev_max_err = -2e10
allowed_err = 1e-5
while flag:
    prev = Z_n.copy()
    for i in range(1,m):
        x = xa + i*dx
        a = [0 for k in range(n-1)]
        b = [0 for k in range(n-1)]
        c = [0 for k in range(n-1)]
        d = [0 for k in range(n-1)]
        for j in range(n-1):
            y = ya + j*dy
            a[j] = 16
            b[j] = -64
            c[j] = 16
            d[j] = x**2 + y**2 -16*Z_n[i-1][j+1] -16*Z_n[i+1][j+1]
            if j == 0:
                d[j] =  d[j] - a[j]*Z_n[i][j]
        d[n-2] -= c[j]*Z_n[i][j+2]
        Z_n[i,1:-1] = thomas_algo(a, b, c, d)
    max_err = abs(Z_n - prev).max()
    if abs(max_err  - prev_max_err) < allowed_err:
        flag = False
    else:
        prev_max_err = max_err


# In[26]:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
wframe = ax.plot_wireframe(xv, yv, Z_n)
plt.show()


# In[ ]:



