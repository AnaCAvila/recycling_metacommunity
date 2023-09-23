
from scipy.optimize import fsolve
import math
import numpy as np
import sympy as sy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal
from numpy import linalg as LA
import decimal



a = [0.9, 0.9, 0.9] #maximum uptake rate
b = [2, 2, 2] #half-saturation (concentration of nutrients at half maximum rate)
m = [0.2, 0.2, 0.2] #mortality rate
I = 0.7 #nutrient input
E = 0.3 #nutrient output
e = [0.5, 0.5, 0.5]    #important values; recycling rates


"""
setting up a range for iterations of the system for each moment t in time
"""
val = 1000
y0 = 1, 1, 1, 1
trange = np.linspace(0, val, val) #equally spaced elements


def system(y, t, a, b, m, I, E, e):
    
    N, R, H, P = y
    dydt = [I-E*N+e[0]*m[0]*R+e[1]*m[1]*H+e[2]*m[2]*P-a[0]*N*R/(b[0] + N),
            a[0]*N*R/(b[0] + N)-m[0]*R-a[1]*H*R/(b[1]+R),
            a[1]*H*R/(b[1]+R)-m[1]*H-a[2]*P*H/(b[2]+H),
            a[2]*P*H/(b[2]+H)-m[2]*P]
    
    return dydt

solved = odeint(system, y0, trange, args = (a, b, m, I, E, e), atol = 1.49012e-10)

"""
setting up a range for iterations of the system for each moment t in time
"""
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


new_P = P_values[int(3*val/4):int(val-1)]
new_H = H_values[int(3*val/4):int(val-1)]
new_R = R_values[int(3*val/4):int(val-1)]
new_N = N_values[int(3*val/4):int(val-1)]


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





###################################### Stability analysis
"""
fixing random errors in the floats to get an equally spaced range
"""

def float_range(start, stop, step):
    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)
 

emin = 0
emax = 1
e_range = float_range(emin, emax, '0.01') #range of r values to be plotted




"""
make dictionaries to store the values of the variables and the corresponding attractors to be plotted
"""

#########################################   r values
"""
when changing variables:
    var_range = q_range
    and change the corresponding values in solved
"""


var_range = e_range

attractors_P = {}
attractors_H = {}
attractors_R = {}
attractors_N = {}


for var in e_range: #iterating through different instances of q and r
    
    iter_e = [var, var, var]
    solved = odeint(system, y0, trange, args = (a, b, m, I, E, iter_e), atol = 1.49012e-10)
    
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
    selecting only values after significant changes have already occurred"""
    
    new_P = P_values[int(3*val/4):int(val-1)]
    new_H = H_values[int(3*val/4):int(val-1)]
    new_R = R_values[int(3*val/4):int(val-1)]
    new_N = N_values[int(3*val/4):int(val-1)]
    
    
    #########measuring stability
    
    
    #find the fixed points
    N, R, H, P = sy.symbols("N R H P")
    
    equilibrium_values = (sy.nsolve([I-E*N+e[0]*m[0]*R+e[1]*m[1]*H+e[2]*m[2]*P-a[0]*N*R/(b[0] + N),
            a[0]*N*R/(b[0] + N)-m[0]*R-a[1]*H*R/(b[1]+R),
            a[1]*H*R/(b[1]+R)-m[1]*H-a[2]*P*H/(b[2]+H),
            a[2]*P*H/(b[2]+H)-m[2]*P],(N, R, H, P),(1,1, 1, 1)) )

    N = equilibrium_values[0]
    R = equilibrium_values[1]
    H = equilibrium_values[2]
    P = equilibrium_values[3]
    


    J = [[-E-a[0]*b[0]*R/((b[0] + N)**2),
      e[0]*m[0]-a[0]*N/(b[0] + N),
      e[1]*m[1],
      e[2]*m[2]],
    
     [a[0]*b[0]*R/((b[0] + N)**2),
      a[0]*N/(b[0] + N)-m[0]-a[1]*b[0]*H/((b[1]+R)**2),
      -a[1]*R/(b[1] + R),
      0],
     [0,
      a[1]*b[1]*H/((b[1] + R)**2),
      a[1]*R/(b[1] + R)-m[1]-a[2]*b[2]*P/((b[2]+H)**2),
      -a[2]*H/(b[2] + H)],   
     [0,
      0,
      a[2]*b[2]*P/((b[2]+H)**2),
      0]]
    
    J = np.array(J, dtype=float)



    eigen = LA.eig(J)
    

    eiglist = []
    for item in eigen[0]:
        eiglist.append(item.real)
        
    ItemCount = 0
    pos_eig_count = 0
    for item in eiglist:
        ItemCount += 1
        if item >= 0:
            pos_eig_count += 1           

        
    if ItemCount == len(eiglist):
        if pos_eig_count == 0:
            print("Stable")
            Stable = True
        else:
            print("Unstable")
            Stable = False
   
            
    ######### finding attractors
    
    
    if Stable == True:
        "select the last point as the stable point and add it to the dictionary at the corresponding variable"
        attractors_P.update({var: solved[-1][3]})
        attractors_H.update({var: solved[-1][2]})
        attractors_R.update({var: solved[-1][1]})
        attractors_N.update({var: solved[-1][0]})

    else:
        # storing the attractors to be added to the dictionary in the future
        attractors_P_list = []
        attractors_H_list = []
        attractors_R_list = []
        attractors_N_list = []
        
        """find maximum and minimum peaks at the later range"""
        
        max_peak_indexes_P, properties = scipy.signal.find_peaks(new_P)
        min_peak_indexes_P, properties = scipy.signal.find_peaks(-new_P)
        max_peak_indexes_H, properties = scipy.signal.find_peaks(new_H)
        min_peak_indexes_H, properties = scipy.signal.find_peaks(-new_H)
        max_peak_indexes_R, properties = scipy.signal.find_peaks(new_R)
        min_peak_indexes_R, properties = scipy.signal.find_peaks(-new_R)
        max_peak_indexes_N, properties = scipy.signal.find_peaks(new_N)
        min_peak_indexes_N, properties = scipy.signal.find_peaks(-new_N)
        
        peak_indexes_P = [max_peak_indexes_P, min_peak_indexes_P]
        peak_indexes_H = [max_peak_indexes_H, min_peak_indexes_H]
        peak_indexes_R = [max_peak_indexes_R, min_peak_indexes_R]
        peak_indexes_N = [max_peak_indexes_N, min_peak_indexes_N]
        
        " max and min peaks are the attractors, and consider rounding up to 0.01 to account only for significant oscillations"
        for index_max in peak_indexes_P[0]:
            peak_max = new_P[index_max]
            rounded_max = '%s' % float('%.1g' % peak_max)
            for index_min in peak_indexes_P[1]:
                peak_min = new_P[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_P_list:
                    attractors_P_list.append(rounded_min)
                if rounded_max not in attractors_P_list:
                    attractors_P_list.append(rounded_max) 
                
        for index_max in peak_indexes_H[0]:
            peak_max = new_H[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_H[1]:
                peak_min = new_H[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_H_list:
                    attractors_H_list.append(rounded_min)
                if rounded_max not in attractors_H_list:
                    attractors_H_list.append(rounded_max)
                    
        for index_max in peak_indexes_R[0]:
            peak_max = new_R[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_R[1]:
                peak_min = new_R[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_R_list:
                    attractors_R_list.append(rounded_min)
                if rounded_max not in attractors_R_list:
                    attractors_R_list.append(rounded_max)
                    
        for index_max in peak_indexes_N[0]:
            peak_max = new_N[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_N[1]:
                peak_min = new_N[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_N_list:
                    attractors_N_list.append(rounded_min)
                if rounded_max not in attractors_N_list:
                    attractors_N_list.append(rounded_max)
        
                    
        "add all values from the list of attractors to the corresponding variable key in the dictionary"
        attractors_P.update({var: attractors_P_list})                        
        attractors_H.update({var: attractors_H_list})
        attractors_R.update({var: attractors_R_list})
        attractors_N.update({var: attractors_N_list})


fig, axs = plt.subplots(2, 2, figsize=(8, 9))



"storing the values in order, and making sure all values stored in y are stored as a float rather than a string"
x = []
y = []
for key in sorted(attractors_P.keys()):
    x.append(key)
    y.append(attractors_P[key])
    
y_ord = [ [] for _ in range(len(attractors_P)) ]
for val in y:
    i=y.index(val)
    for peak in val:
        y_ord[i].append(float(peak))


for xe, ye in zip(x, y_ord):
    axs[1, 1].set_title("Predator")
    axs[1, 1].scatter([xe] * len(ye), ye) 
    axs[1, 1].set_xlabel("r")
    max_yticks = 10
    yloc = plt.MaxNLocator(max_yticks)
    axs[1, 1].yaxis.set_major_locator(yloc)
    fig.suptitle('Bifurcation plots - max recycling rate') 
    
    
    
    
x1 = []
y1 = []
for key in sorted(attractors_H.keys()):
    x1.append(key)
    y1.append(attractors_H[key])  
    
y1_ord = [ [] for _ in range(len(attractors_H)) ]
for val in y1:
    i=y1.index(val)
    for peak in val:
        y1_ord[i].append(float(peak))

for xe, ye in zip(x1, y1_ord):
    axs[0, 1].set_title("Herbivore")
    axs[0, 1].scatter([xe] * len(ye), ye) 
    axs[0, 1].set_xlabel("r")
        
    
    
    
x2 = []
y2 = []
for key in sorted(attractors_R.keys()):
    x2.append(key)
    y2.append(attractors_R[key])  
y2_ord = [ [] for _ in range(len(attractors_R)) ]
for val in y2:
    i=y2.index(val)
    for peak in val:
        y2_ord[i].append(float(peak))  
for xe, ye in zip(x2, y2_ord):
    axs[0, 0].set_title("Producers")
    axs[0, 0].scatter([xe] * len(ye), ye) 
    axs[0, 0].set_xlabel("r")




x3 = []
y3 = [] 
for key in sorted(attractors_N.keys()):
    x3.append(key)
    y3.append(attractors_N[key])  
y3_ord = [ [] for _ in range(len(attractors_N)) ]
for val in y3:
    i=y3.index(val)
    for peak in val:
        y3_ord[i].append(float(peak))  
for xe, ye in zip(x3, y3_ord):
    axs[1, 0].set_title("Nutrients")
    axs[1, 0].scatter([xe] * len(ye), ye) 
    axs[1, 0].set_xlabel("r")
    