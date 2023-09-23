"""
3 ecosystem levels, two patches, with recycling and delay
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal
from numpy import linalg as LA


a = [0.45] #maximum uptake rate
b = [0.5] #half-saturation (concentration of nutrients at half maximum rate)
m = [0.01] #mortality rate
I = 0.7 #nutrient input
E = 0.3 #nutrient output

z = [0.6] #maximum recycling rate
l = [0] #deforestation rate (biomass loss)
q = [1] #Hill coefficient - ease of decomposition of the matter. This can vary through the season.
s = [1] #half-saturation (concentration of dead organic matter (delta*R) at half maximum rate)



val = 1000
y0 = 1, 1, 1, 1
t = np.linspace(0, val, 1000) #equally spaced elements

def system(y, t, a, b, m, I, E, r, l, q, s):
    N, R, H, P = y
    dydt = [I - E*N + (r[0]*(m[0]*R)**q[0])/(s[0]**q[0]+(m[0]*R)**q[0]) - a[0]*N*R/(b[0] + N),
            
            a[0]*N*R/(b[0] + N) - m[0]*R - l[0]*R]

    return dydt


#for the stability analysis, look at the stable point per each equation:
    
def dN(yn, t, a, b, m, I, E, r, l, q, s):
    N = yn
    N_dydt = [I - E*N + (r[0]*(m[0]*R)**q[0])/(s[0]**q[0]+(m[0]*R)**q[0]) - a[0]*N*R/(b[0] + N)]

    return N_dydt


def dR(yr, t, a, b, m, I, E, r, l, q, s):
    R = yr
    R_dydt = [a[0]*N*R/(b[0] + N) - m[0]*R - l[0]*R]

    return R_dydt



#gives the equilibrium values, not if it's stable
r = odeint(system, y0, t, args = (a, b, m, I, E, z, l, q, s), atol = 1.49012e-10)
#does the ODE and iterates

plt.plot(t, r[:, 0], 'b', label='N(t)')
plt.plot(t, r[:, 1], 'g', label='R(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()


N = r[val-1,0] #final N value, if stable
R = r[val-1,1] #final R value, if stable


############################################################



J = [[-E-(a[0]*b[0]*R)/(b[0]+N)**2, (np.power(m[0], q[0])*q[0]*z[0]*np.power((m[0]*R), q[0]))/(R*(np.power((m[0]*R), q[0])+np.power(m[0], q[0]))**2) - a[0]*N/(b[0]+N)],
        [(a[0]*b[0]*R)/(b[0]+N)**2, a[0]*N/(b[0]+N)-m[0]-l[0]]]


       
s = LA.eig(J)
print(s[0])

eiglist = []
for item in s[0]:
    eiglist.append(item.real)
    
    
print(eiglist)


ItemCount = 0
Stable = True
for item in eiglist:
    ItemCount += 1
    if item >=0:
        Stable == False
if ItemCount == len(eiglist):
    if Stable == True:
        print("System is stable!")
        
       
########################################  Bifurcation plots

