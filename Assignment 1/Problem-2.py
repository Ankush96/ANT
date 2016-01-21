import matplotlib.pyplot as plt
import numpy as np

def f1(x, p ,q):
    return q

def f2(x, p, q):
    return p

def runge_kutta_4(x, p, q, h):
    u = np.vstack((p,q))
    
    for i in range(len(p)-1):
        k1 = np.array((f1(x[i], p[i], q[i]), f2(x[i], p[i], q[i])))
        k2 = np.array((f1(x[i] + 0.5*h, p[i] + k1[0]*h*0.5, q[i] + k1[1]*h*0.5 ), f2(x[i] + 0.5*h, p[i] + k1[0]*h*0.5, q[i] + k1[1]*h*0.5 )))
        k3 = np.array((f1(x[i] + 0.5*h, p[i] + k2[0]*h*0.5, q[i] + k2[1]*h*0.5 ), f2(x[i] + 0.5*h, p[i] + k2[0]*h*0.5, q[i] + k2[1]*h*0.5 )))
        k4 = np.array((f1(x[i] + h, p[i] + k3[0]*h, q[i] + k3[1]*h ), f2(x[i] + h, p[i] + k3[0]*h, q[i] + k3[1]*h )))
        u[:,i+1] = u[:,i] + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        p = u[0]
        q = u[1]
    return p,q

def bvp(ya, yb, a0, a1, start, end, h, conv_threshold):
    x = np.linspace(start, end, int((end - start)/(1.0 * h) +1) )
    p = x.copy()
    q = x.copy()
    p[0] = ya   
    
    while abs(a0 - a1) > conv_threshold:
        q[0] = a0
        p, q = runge_kutta_4(x, p, q, h)
        y0 = p[-1]
        q[0] = a1
        p, q = runge_kutta_4(x, p , q, h)
        y1 = p[-1]
        
        a2 = a0 - (a1 - a0)*(y0 - yb)/(1.0*(y1 - y0))
        a0 = a1
        a1 = a2
    print "Alpha has converged to %s" % (a0)
    print '\nx is \t %s' % x
    print 'y is \t %s' % p
    print 'dy/dx is %s' % q
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot(x, p)
    plt.show()
    
def main():
    bvp(0, -1, 0.3, 0.4, 0, 1, 0.1, 0.0001)  

main()
