"""
3 ecosystem levels, two patches, no delay (just recycling)
"""

from scipy.optimize import fsolve
import math
import numpy as np
import sympy as sy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal
from numpy import linalg as LA
import decimal


a = [0.9, 0.9, 0.9]
b = [2, 2, 2]
m = [0.1, 0.2, 0.2]
e = [0.1, 0.1, 0.9]    #important values; recycling rates
d = [0.5, 0.5, 0.5, 0.5]
I = 0.7
E = 0.3

"""
setting up a range for iterations of the system for each moment t in time
"""

# enter initial conditions

val = 1000
y0 = 1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8
trange = np.linspace(0, val, 1000)

def f(y, trange, a, b, m, e, d, I, E):
    N, R, H, P, N2, R2, H2, P2 = y
    dydt = [I-E*N+e[0]*m[0]*R+e[1]*m[1]*H+e[2]*m[2]*P-a[0]*N*R/(b[0] + N)+d[0]*(N2-N),
            a[0]*N*R/(b[0] + N)-m[0]*R-a[1]*H*R/(b[1]+R)+d[1]*(R2-R),
            a[1]*H*R/(b[1]+R)-m[1]*H-a[2]*P*H/(b[2]+H)+d[2]*(H2-H),
            a[2]*P*H/(b[2]+H)-m[2]*P+d[3]*(P2-P),
            I-E*N2+e[0]*m[0]*R2+e[1]*m[1]*H2+e[2]*m[2]*P2-a[0]*N2*R2/(b[0] + N2)+d[0]*(N-N2),
            a[0]*N2*R2/(b[0] + N2)-m[0]*R2-a[1]*H2*R2/(b[1]+R2)+d[1]*(R-R2),
            a[1]*H2*R2/(b[1]+R2)-m[1]*H2-a[2]*P2*H2/(b[2]+H2)+d[2]*(H-H2),
            a[2]*P2*H2/(b[2]+H2)-m[2]*P2+d[3]*(P-P2)]
    return dydt


solved = odeint(f, y0, trange, args = (a, b, m, e, d, I, E), atol = 1.49012e-10)


"""
defining the population array per level
"""


N1_values = np.zeros(val)
R1_values = np.zeros(val)
H1_values = np.zeros(val)
P1_values = np.zeros(val)
N2_values = np.zeros(val)
R2_values = np.zeros(val)
H2_values = np.zeros(val)
P2_values = np.zeros(val)

i=0
for value in solved:
    N1_values[i] = value[0]
    R1_values[i] = value[1]
    H1_values[i] = value[2]
    P1_values[i] = value[3]
    N2_values[i] = value[4]
    R2_values[i] = value[5]
    H2_values[i] = value[6]
    P2_values[i] = value[7]    
    i+=1
    
    
#selecting only the last iterations for further analysis
new_P1 = P1_values[int(3*val/4):int(val-1)]
new_H1 = H1_values[int(3*val/4):int(val-1)]
new_R1 = R1_values[int(3*val/4):int(val-1)]
new_N1 = N1_values[int(3*val/4):int(val-1)]    
new_P2 = P2_values[int(3*val/4):int(val-1)]
new_H2 = H2_values[int(3*val/4):int(val-1)]
new_R2 = R2_values[int(3*val/4):int(val-1)]
new_N2 = N2_values[int(3*val/4):int(val-1)]      
    
    
    
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


plt.plot(trange, solved[:, 4], 'b', label='N2(t)')
plt.plot(trange, solved[:, 5], 'g', label='R2(t)')
plt.plot(trange, solved[:, 6], 'y', label='H2(t)')
plt.plot(trange, solved[:, 7], 'r', label='P2(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()




########################################    Stability analysis

"""
fixing random errors in the floats to get an equally spaced range
"""

def float_range(start, stop, step):
    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)
 

rmin = 0
rmax = 1
r_range = float_range(rmin, rmax, '0.01') #range of r values to be plotted

"""
make dictionaries to store the values of the variables and the corresponding attractors to be plotted
"""

var_range = r_range

attractors_P2 = {}
attractors_H2 = {}
attractors_R2 = {}
attractors_N2 = {}
attractors_P1 = {}
attractors_H1 = {}
attractors_R1 = {}
attractors_N1 = {}




for var in r_range: #iterating through different instances of r
    
    iter_r = [var, var, var, var]
    solved = odeint(f, y0, trange, args = (a, b, m, iter_r, d, I, E), atol = 1.49012e-10)

    
    N1_values = np.zeros(val)
    R1_values = np.zeros(val)
    H1_values = np.zeros(val)
    P1_values = np.zeros(val)
    N2_values = np.zeros(val)
    R2_values = np.zeros(val)
    H2_values = np.zeros(val)
    P2_values = np.zeros(val)
    
    i=0
    for value in solved:
        N1_values[i] = value[0]
        R1_values[i] = value[1]
        H1_values[i] = value[2]
        P1_values[i] = value[3]
        N2_values[i] = value[4]
        R2_values[i] = value[5]
        H2_values[i] = value[6]
        P2_values[i] = value[7]    
        i+=1
        
        
    #selecting only the last iterations for further analysis
    new_P1 = P1_values[int(3*val/4):int(val-1)]
    new_H1 = H1_values[int(3*val/4):int(val-1)]
    new_R1 = R1_values[int(3*val/4):int(val-1)]
    new_N1 = N1_values[int(3*val/4):int(val-1)]    
    new_P2 = P2_values[int(3*val/4):int(val-1)]
    new_H2 = H2_values[int(3*val/4):int(val-1)]
    new_R2 = R2_values[int(3*val/4):int(val-1)]
    new_N2 = N2_values[int(3*val/4):int(val-1)]  


    
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
    



    C = np.array([[-1,1],[1,-1]])
    w, v = LA.eig(C)
    
    
    
    diffusion_matrix = [[d[0],0,0,0],[0,d[1],0,0],[0,0,d[2],0],[0,0,0,d[3]]]
    
    ###connectivity
    
    
    
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

    #how does the Jacobian work for two patches?
    
    V = np.zeros((4,4)) #create a storage matrix with zeroes inside 
    
    
    for i in range(0, 3):
        for j in range (0, 3):
            #the first C eigenvalue is 0 so the first V matrix is just equal to J
            V[i][j] = J[i][j] + (diffusion_matrix[i][j]*-2)
    
    
    t, t1 = LA.eig(V)
    s, s1 = LA.eig(J)
    s = s.tolist()
    t = t.tolist()
    eiglist = []
    for item in t:
        eiglist.append(item.real)
   # for item in s:
    #    eiglist.append(item.real)
    
    
    ItemCount = 0
    Stable = True
    for item in eiglist:
        ItemCount += 1
        if item >=0:
            Stable == False
            print("System is NOT stable!")
    if ItemCount == len(eiglist):
        if Stable == True:
            print("System is stable!")
            
            
    ######### finding attractors
    
    
    
    if Stable == True:
        "select the last point as the stable point and add it to the dictionary at the corresponding variable"
        attractors_P1.update({var: solved[-1][3]})
        attractors_H1.update({var: solved[-1][2]})
        attractors_R1.update({var: solved[-1][1]})
        attractors_N1.update({var: solved[-1][0]})
        attractors_P2.update({var: solved[-1][7]})
        attractors_H2.update({var: solved[-1][6]})
        attractors_R2.update({var: solved[-1][5]})
        attractors_N2.update({var: solved[-1][4]})

    else:
        # storing the attractors to be added to the dictionary in the future
        attractors_P2_list = []
        attractors_H2_list = []
        attractors_R2_list = []
        attractors_N2_list = []
        attractors_P1_list = []
        attractors_H1_list = []
        attractors_R1_list = []
        attractors_N1_list = []
        
        """find maximum and minimum peaks at the later range  -  patch 1"""
        
        max_peak_indexes_P1, properties = scipy.signal.find_peaks(new_P1)
        min_peak_indexes_P1, properties = scipy.signal.find_peaks(-new_P1)
        max_peak_indexes_H1, properties = scipy.signal.find_peaks(new_H1)
        min_peak_indexes_H1, properties = scipy.signal.find_peaks(-new_H1)
        max_peak_indexes_R1, properties = scipy.signal.find_peaks(new_R1)
        min_peak_indexes_R1, properties = scipy.signal.find_peaks(-new_R1)
        max_peak_indexes_N1, properties = scipy.signal.find_peaks(new_N1)
        min_peak_indexes_N1, properties = scipy.signal.find_peaks(-new_N1)
        
        peak_indexes_P1 = [max_peak_indexes_P1, min_peak_indexes_P1]
        peak_indexes_H1 = [max_peak_indexes_H1, min_peak_indexes_H1]
        peak_indexes_R1 = [max_peak_indexes_R1, min_peak_indexes_R1]
        peak_indexes_N1 = [max_peak_indexes_N1, min_peak_indexes_N1]
        
        " max and min peaks are the attractors, and consider rounding up to 0.01 to account only for significant oscillations"
        for index_max in peak_indexes_P1[0]:
            peak_max = new_P1[index_max]
            rounded_max = '%s' % float('%.1g' % peak_max)
            for index_min in peak_indexes_P1[1]:
                peak_min = new_P1[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_P1_list:
                    attractors_P1_list.append(rounded_min)
                if rounded_max not in attractors_P1_list:
                    attractors_P1_list.append(rounded_max) 
                
        for index_max in peak_indexes_H1[0]:
            peak_max = new_H1[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_H1[1]:
                peak_min = new_H1[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_H1_list:
                    attractors_H1_list.append(rounded_min)
                if rounded_max not in attractors_H1_list:
                    attractors_H1_list.append(rounded_max)
                    
        for index_max in peak_indexes_R1[0]:
            peak_max = new_R1[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_R1[1]:
                peak_min = new_R1[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_R1_list:
                    attractors_R1_list.append(rounded_min)
                if rounded_max not in attractors_R1_list:
                    attractors_R1_list.append(rounded_max)
                    
        for index_max in peak_indexes_N1[0]:
            peak_max = new_N1[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_N1[1]:
                peak_min = new_N1[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_N1_list:
                    attractors_N1_list.append(rounded_min)
                if rounded_max not in attractors_N1_list:
                    attractors_N1_list.append(rounded_max)
        
                    
        "add all values from the list of attractors to the corresponding variable key in the dictionary"
        attractors_P1.update({var: attractors_P1_list})                        
        attractors_H1.update({var: attractors_H1_list})
        attractors_R1.update({var: attractors_R1_list})
        attractors_N1.update({var: attractors_N1_list})


        
        """find maximum and minimum peaks at the later range  -  patch 2"""
        
        max_peak_indexes_P2, properties = scipy.signal.find_peaks(new_P2)
        min_peak_indexes_P2, properties = scipy.signal.find_peaks(-new_P2)
        max_peak_indexes_H2, properties = scipy.signal.find_peaks(new_H2)
        min_peak_indexes_H2, properties = scipy.signal.find_peaks(-new_H2)
        max_peak_indexes_R2, properties = scipy.signal.find_peaks(new_R2)
        min_peak_indexes_R2, properties = scipy.signal.find_peaks(-new_R2)
        max_peak_indexes_N2, properties = scipy.signal.find_peaks(new_N2)
        min_peak_indexes_N2, properties = scipy.signal.find_peaks(-new_N2)
        
        peak_indexes_P2 = [max_peak_indexes_P2, min_peak_indexes_P2]
        peak_indexes_H2 = [max_peak_indexes_H2, min_peak_indexes_H2]
        peak_indexes_R2 = [max_peak_indexes_R2, min_peak_indexes_R2]
        peak_indexes_N2 = [max_peak_indexes_N2, min_peak_indexes_N2]
        
        " max and min peaks are the attractors, and consider rounding up to 0.01 to account only for significant oscillations"
        for index_max in peak_indexes_P2[0]:
            peak_max = new_P2[index_max]
            rounded_max = '%s' % float('%.1g' % peak_max)
            for index_min in peak_indexes_P2[1]:
                peak_min = new_P2[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_P2_list:
                    attractors_P2_list.append(rounded_min)
                if rounded_max not in attractors_P2_list:
                    attractors_P2_list.append(rounded_max) 
                
        for index_max in peak_indexes_H2[0]:
            peak_max = new_H2[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_H2[1]:
                peak_min = new_H2[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_H2_list:
                    attractors_H2_list.append(rounded_min)
                if rounded_max not in attractors_H2_list:
                    attractors_H2_list.append(rounded_max)
                    
        for index_max in peak_indexes_R2[0]:
            peak_max = new_R2[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_R2[1]:
                peak_min = new_R2[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_R2_list:
                    attractors_R2_list.append(rounded_min)
                if rounded_max not in attractors_R2_list:
                    attractors_R2_list.append(rounded_max)
                    
        for index_max in peak_indexes_N2[0]:
            peak_max = new_N2[index_max]
            rounded_max = '%s' % float('%.01g' % peak_max)
            for index_min in peak_indexes_N2[1]:
                peak_min = new_N2[index_min]
                rounded_min = '%s' % float('%.01g' % peak_min)
                if rounded_min not in attractors_N2_list:
                    attractors_N2_list.append(rounded_min)
                if rounded_max not in attractors_N2_list:
                    attractors_N2_list.append(rounded_max)
        
                    
        "add all values from the list of attractors to the corresponding variable key in the dictionary"
        attractors_P2.update({var: attractors_P2_list})                        
        attractors_H2.update({var: attractors_H2_list})
        attractors_R2.update({var: attractors_R2_list})
        attractors_N2.update({var: attractors_N2_list})



###patch 1


fig, axs = plt.subplots(2, 2, figsize=(8, 9))



"storing the values in order, and making sure all values stored in y are stored as a float rather than a string"
x = []
y = []
for key in sorted(attractors_P1.keys()):
    x.append(key)
    y.append(attractors_P1[key])
    
y_ord = [ [] for _ in range(len(attractors_P1)) ]
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
for key in sorted(attractors_H1.keys()):
    x1.append(key)
    y1.append(attractors_H1[key])  
    
y1_ord = [ [] for _ in range(len(attractors_H1)) ]
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
for key in sorted(attractors_R1.keys()):
    x2.append(key)
    y2.append(attractors_R1[key])  
y2_ord = [ [] for _ in range(len(attractors_R1)) ]
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
for key in sorted(attractors_N1.keys()):
    x3.append(key)
    y3.append(attractors_N1[key])  
y3_ord = [ [] for _ in range(len(attractors_N1)) ]
for val in y3:
    i=y3.index(val)
    for peak in val:
        y3_ord[i].append(float(peak))  
for xe, ye in zip(x3, y3_ord):
    axs[1, 0].set_title("Nutrients")
    axs[1, 0].scatter([xe] * len(ye), ye) 
    axs[1, 0].set_xlabel("r")
    
            
    

###patch 2




fig, axs = plt.subplots(2, 2, figsize=(8, 9))



"storing the values in order, and making sure all values stored in y are stored as a float rather than a string"
x = []
y = []
for key in sorted(attractors_P2.keys()):
    x.append(key)
    y.append(attractors_P2[key])
    
y_ord = [ [] for _ in range(len(attractors_P2)) ]
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
for key in sorted(attractors_H2.keys()):
    x1.append(key)
    y1.append(attractors_H2[key])  
    
y1_ord = [ [] for _ in range(len(attractors_H2)) ]
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
for key in sorted(attractors_R2.keys()):
    x2.append(key)
    y2.append(attractors_R2[key])  
y2_ord = [ [] for _ in range(len(attractors_R2)) ]
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
for key in sorted(attractors_N2.keys()):
    x3.append(key)
    y3.append(attractors_N2[key])  
y3_ord = [ [] for _ in range(len(attractors_N2)) ]
for val in y3:
    i=y3.index(val)
    for peak in val:
        y3_ord[i].append(float(peak))  
for xe, ye in zip(x3, y3_ord):
    axs[1, 0].set_title("Nutrients")
    axs[1, 0].scatter([xe] * len(ye), ye) 
    axs[1, 0].set_xlabel("r")
    
