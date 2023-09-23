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


for var in e_range: #iterating through different instances of r
    
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
    



########################################    Stability analysis


'''last values of N and R outputted by the function'''
N = solved[val-1,0]
R = solved[val-1,1]
H = solved[val-1,2]
P = solved[val-1,3]
N2 = solved[val-1,4]
R2 = solved[val-1,5]
H2 = solved[val-1,6]
P2 = solved[val-1,7]




C = np.array([[-1,1],[1,-1]])
w, v = LA.eig(C)


V = np.zeros((4,4))


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
for item in s:
    eiglist.append(item.real)


ItemCount = 0
Stable = True
for item in eiglist:
    ItemCount += 1
    if item >=0:
        Stable == False
if ItemCount == len(eiglist):
    if Stable == True:
        print("System is stable!")
        

           
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
    