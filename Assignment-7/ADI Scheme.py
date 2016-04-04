import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

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

dx = dy = 0.05
r = 1.0/6
dt = r*dx**2
xa = ya = 0
xb = yb = 1

m = n = int(1.0/dx)
x_grid = np.linspace(xa,xb,n+1)
y_grid = np.linspace(ya,yb,m+1)
yv, xv = np.meshgrid(x_grid, y_grid)

Z_n,_ = np.meshgrid(np.zeros(len(x_grid)),np.zeros(len(y_grid)))


def step1(j):
    a = [0 for i in range(n-1)]
    b = [0 for i in range(n-1)]
    c = [0 for i in range(n-1)]
    d = [0 for i in range(n-1)]
    
    for i in range(n-1):
        # As i starts from 0, we define x = l + (i+1)*h
        x = xa+(i+1)*dx  
        y = ya + j*dy
        a[i] = r
        b[i] = -2.0*(1+r)
        c[i] = r
        d[i] = -1*r*Z_n[i+1][j+1] +2*(r-1)*Z_n[i+1][j] -1*r*Z_n[i+1][j-1]
        if i == 0:
            d[i] =  d[i] - a[i] * np.exp(0.2*np.pi*xa)*np.sin(0.2*np.pi*y)

    d[n-2] = d[n-2] -  c[n-2] * np.exp(0.2*np.pi*xb)*np.sin(0.2*np.pi*y)

    return thomas_algo(a, b, c, d)

def step2(i):
    a = [0 for j in range(m-1)]
    b = [0 for j in range(m-1)]
    c = [0 for j in range(m-1)]
    d = [0 for j in range(m-1)]
    
    for j in range(m-1):
        # As i starts from 0, we define x = l + (i+1)*h
        y = ya+(j+1)*dy  
        x = xa + i*dx
        a[j] = r
        b[j] = -2.0*(1+r)
        c[j] = r
        d[j] = -1*r*Z_n[i-1][j+1] +2*(r-1)*Z_n[i][j+1] -1*r*Z_n[i+1][j+1]
        if j == 0:
            d[j] =  d[j] - a[j] * np.exp(0.2*np.pi*x)*np.sin(0.2*np.pi*ya)

    d[m-2] = d[m-2] -  c[m-2] * np.exp(0.2*np.pi*x)*np.sin(0.2*np.pi*yb)

    return thomas_algo(a, b, c, d)

def adi():
    new_Z,_ = np.meshgrid(np.zeros(len(x_grid)),np.zeros(len(y_grid)))
    new_Z[0,:] = np.exp(0.2*np.pi*xa)*np.sin(0.2*np.pi*y_grid)
    new_Z[-1,:] = np.exp(0.2*np.pi*xb)*np.sin(0.2*np.pi*y_grid)
    new_Z[:,0] = np.exp(0.2*np.pi*x_grid)*np.sin(0.2*np.pi*ya)
    new_Z[:,-1] = np.exp(0.2*np.pi*x_grid)*np.sin(0.2*np.pi*yb)
    for j in range(1,n):
        v_j = step1(j)
        new_Z[1:-1,j] =  v_j
    Z_n = new_Z
    for i in range(1,m):
        v_i = step2(i)
        new_Z[i,1:-1] =  v_i
    Z_n = new_Z
    return Z_n

def generate(X, Y, t):
    if t==0:
        p,_ = np.meshgrid(np.zeros(len(x_grid)),np.zeros(len(y_grid)))
        return p
    else:
        return adi()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Z_n = generate(xv,yv,0)

wframe = None
for i in range(100):

    oldcol = wframe
    for j in range(int(0.2*i**2)):
        Z_n = generate(xv, yv, 1)
    wframe = ax.plot_wireframe(xv, yv, Z_n)

    # Remove old line collection before drawing
    if oldcol is not None:
        ax.collections.remove(oldcol)

    plt.pause(.05)
plt.show()


