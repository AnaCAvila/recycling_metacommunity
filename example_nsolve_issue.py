
"""
Three trophic level nutrient-autotroph model
added delay in recycling and l term for deforestation
"""

from scipy.optimize import fsolve
import math
import numpy as np
import sympy as sy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal
from numpy import linalg as LA



a = [0.9, 0.9, 0.9] #maximum uptake rate
b = [2, 2, 2] #half-saturation (concentration of nutrients at half maximum rate)
m = [0.2, 0.2, 0.2] #mortality rate
I = 0.7 #nutrient input
E = 0.3 #nutrient output
r = [0.15] #maximum recycling rate
l = [0] #deforestation rate (biomass loss)
q = [1] #Hill coefficient - ease of decomposition of the matter. This can vary through the season.
s = [0.9] #half-saturation (concentration of dead organic matter (delta*R) at half maximum rate)


"""
setting up a range for iterations of the system for each moment t in time
"""
val = 1000
y0 = 1, 1, 1, 1
trange = np.linspace(0, val, val) #equally spaced elements


def system(y, t, a, b, m, I, E, r, l, q, s):
    
    N, R, H, P = y
    dydt = [I - E*N + (r[0]*(m[0]*R)**q[0])/(s[0]**q[0]+(m[0]*R)**q[0]) + (r[0]*(m[1]*N)**q[0])/(s[0]**q[0]+(m[1]*N)**q[0]) + (r[0]*(m[2]*P)**q[0])/(s[0]**q[0]+(m[2]*P)**q[0]) - a[0]*N*R/(b[0] + N),
            
            a[0]*N*R/(b[0] + N) - m[0]*R - a[1]*H*R/(b[1] + R),
            
            a[1]*H*R/(b[1] + R) - m[1]*H - a[1]*H*P/(b[2] + H),
            
            a[1]*H*P/(b[2] + H) - m[2]*P]
    
    return dydt


solved = odeint(system, y0, trange, args = (a, b, m, I, E, r, l, q, s), atol = 1.49012e-10)


N_values = np.zeros(val)
R_values = np.zeros(val)
H_values = np.zeros(val)
P_values = np.zeros(val)

i=0
for value in solved:
    N_values[i] = value[0]
    R_values[i] = value[1]
    H_values[i] = value[2]
    P_values[i] = value[3]
    i+=1


"""
Plotting dynamics
"""
plt.plot(trange, solved[:, 0], 'b', label='N(t)')
plt.plot(trange, solved[:, 1], 'g', label='R(t)')
plt.plot(trange, solved[:, 2], 'y', label='H(t)')
plt.plot(trange, solved[:, 3], 'r', label='P(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()


N = solved[val-1,0]
R = solved[val-1,1]
H = solved[val-1,2]
P = solved[val-1,3]


print(N, R, H, P)


N, R, H, P = sy.symbols("N R H P")




equilibrium_values = (sy.nsolve([I - E*N + (r[0]*(m[0]*R)**q[0])/(s[0]**q[0]+(m[0]*R)**q[0]) - a[0]*N*R/(b[0] + N),
            
            a[0]*N*R/(b[0] + N) - m[0]*R - l[0]*R - a[1]*H*R/(b[1] + R),
            
            a[1]*H*R/(b[1] + R) - m[1]*H - a[1]*H*P/(b[2] + H),
            
            a[1]*H*P/(b[2] + H) - m[2]*P],(N, R, H, P),(1,1, 1, 1)) )

print(equilibrium_values)



