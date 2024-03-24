# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob
from scipy.stats import skewnorm
from ipywidgets import interactive
# %%
def find_mean(distribution) :
    distribution = distribution / np.sum(distribution)
    
    mean_value = np.sum(np.arange(0,len(distribution),1)*distribution)
    
    return mean_value

def find_median(distribution):
    
    # Calculate the cumulative sum of probabilities
    cumulative_probs = np.cumsum(distribution)
    
    # Find the index where cumulative sum is closest to half of total sum
    median_index = np.abs(cumulative_probs - np.sum(distribution) / 2).argmin()
    
    # Return the value at median index
    return median_index

def rot_diff_par(l,AR) :
    
    eta = 2*10**(-1)	  #1.61*10^(-2);
    kT = 4.11*10**(-21)
    S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
    A = (3*kT)/(np.pi*eta*l**3)
    G = (0.5*AR**2*( ((2*AR**2-1) / AR)*S-1) ) / (AR**2-1)
    D = A*G

    return D

file_path = 'C:\\Users\\marc3\\OneDrive\\Documents\\INTERNSHIP M2\\TEM\\B1-MB03 MARCELLO\\measurements\\aspectRATIO.txt'  # replace with the actual file path

dat = np.loadtxt(file_path,  delimiter='\t', usecols=(4), unpack=True)

dat = np.abs(dat)

l = []
w = []
for i in range(0,len(dat),2):
    l.append(dat[i])
for i in range(0-1,len(dat),2):
    w.append(dat[i])

x1 = np.arange(0,len(l)) 
x2 = np.arange(0,2*len(w),2) 

AR = []
for i in range(len(l)):
    AR.append(l[i]/w[i])

l = np.array(l)
AR = np.array(AR)
#plt.plot(AR, label = "AR")
#plt.show()



eta = 2*10**(-2)	  #1.61*10^(-2);
kT = 4.11*10**(-21)
l = 0.000001*l  #conversion to m from um

l = 10**9*l # l in nm


def lin(x,m,q):
    return m*x+q

qm, qmpcov = curve_fit(lin,l,AR)

def ar(l):
    aspect_ratio = qm[0]*l+qm[1]
    return 20

def ar_max(l):
    aspect_ratio = (qm[0]+np.sqrt(qmpcov[0,0])) * l + qm[1]+np.sqrt(qmpcov[1,1])
    return aspect_ratio 

def ar_min(l):
    aspect_ratio = (qm[0]-np.sqrt(qmpcov[0,0])) * l -np.sqrt(qmpcov[1,1])
    return aspect_ratio 

plot_lAR = False
if plot_lAR == True :
    plt.scatter(l,AR)
    plt.plot(np.arange(min(l),max(l)), ar(np.arange(min(l),max(l))) ,color='r',label = f'{qm[0]}  {qm[1]}' )
    plt.plot(np.arange(min(l),max(l)), ar_max(np.arange(min(l),max(l))) ,color='b')
    plt.plot(np.arange(min(l),max(l)), ar_min(np.arange(min(l),max(l))) ,color='g')
    plt.legend()
    plt.xlabel('legths [nm]')
    plt.ylabel('AR')
    plt.show()

print(f'AR = {qm[0]}l {qm[1]}')

# %%
# Create histogram
hist, bins = np.histogram(l, bins=30, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Define a skewed normal distribution function
def skew_normal_pdf(x, loc, scale, skew):
    return skewnorm.pdf(x, a=skew, loc=loc, scale=scale)*skew*x

# Fit the data to the skewed normal distribution
popt, pcov = curve_fit(skew_normal_pdf, bin_centers, hist, p0=[np.mean(l), np.std(l), 0])

def correction(lengths, loc, scale, skew):
    Surface =  2* np.pi*lengths/ar(lengths)*lengths
    #V = np.pi*( (0.5*lengths/ar(lengths))**2)*lengths
    new_dist = Surface*skewnorm.pdf(lengths, a=skew, loc=loc, scale=scale)*skew*lengths
    return new_dist
#%%
# Plot the fitted curve
x_range = np.linspace(min(l), max(l), 100)
dist = skew_normal_pdf(x_range, *popt)
eff_dist = correction(x_range, *popt)
#eff_dist =  np.pi*((0.5*ar(x_range)/x_range)**2)*x_range*skew_normal_pdf(x_range, *popt)

plt.hist(l, bins=30, density=True, alpha=0.5, label='lengths distribution')
plt.plot(x_range, dist, color = 'b', label='fitted distribution')
plt.plot(x_range, eff_dist/(3.5*sum(eff_dist)), color = 'r', label='fitted distribution*Vol')
plt.axvline(  x = np.argmax(eff_dist) * (max(l)-min(l))/100 +min(l), color = 'r', linestyle='--', alpha = 0.7 )
plt.axvline(  x = np.argmax( skew_normal_pdf(x_range, *popt) ) * (max(l)-min(l))/100 +min(l), color = 'b', linestyle='--', alpha = 0.7 )
plt.xlabel('length [nm]')
plt.legend()
plt.show()

# %%
ldist = 1500
dist = skew_normal_pdf(np.arange(20,ldist,1), *popt)
eff_dist = correction(np.arange(20,ldist,1), *popt)
print('>>> MODE :')
effective_l_mode = np.argmax( eff_dist ) + 20
effective_l_mode = effective_l_mode*10**(-9)

mode_l = np.argmax( dist ) +20
mode_l = mode_l*10**(-9)
effective_D_mode = rot_diff_par(effective_l_mode,ar(mode_l*10**9)) #( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*effective_l_mode)**3)
calculated_D_mode = rot_diff_par(mode_l,ar(mode_l*10**9)) #( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*mode_l)**3)

print('effective l[nm] = ',effective_l_mode*10**9)
print('calculated l[nm] = ',mode_l*10**9)

print('effective Diff coeff = ',effective_D_mode)
print('calculated Ddiif coeff = ',calculated_D_mode)

print('.......................................................')


print('>>> MEDIAN :')
effective_l_median = find_median(eff_dist)+20 #np.abs(eff_dist - find_median(eff_dist)).argmin()
effective_l_median = effective_l_median*10**(-9) #trsform to m for D calculation

median_l = find_median(dist)+20
median_l = median_l*10**(-9)

effective_D_median = rot_diff_par(effective_l_median,ar(median_l*10**9))#( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*effective_l_median)**3)
calculated_D_median = rot_diff_par(median_l,ar(median_l*10**9))#( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*median_l)**3)

print('effective l[nm] = ',effective_l_median*10**9) #trasform to nm for clarity
print('calculated l[nm] = ',median_l*10**9)

print('effective Diff coeff = ',effective_D_median)
print('calculated Diif coeff = ',calculated_D_median)

print('.......................................................')

print('>>> MEAN/AVERAGE :')
effective_l_average = find_mean(eff_dist)+20
effective_l_average = effective_l_average*10**(-9)

average_l = find_mean(dist)+20
average_l = average_l*10**(-9)

effective_D_average = rot_diff_par(effective_l_average,ar(average_l*10**9))#( 3*kT*(-1+2*np.log(np.mean(2*AR))) )/(16*np.pi*eta*(0.5*effective_l_average)**3)
calculated_D_average = rot_diff_par(average_l,ar(average_l*10**9)) #( 3*kT*(-1+2*np.log(np.mean(2*AR))) )/(16*np.pi*eta*(0.5*average_l)**3)



print('effective l[nm] = ',effective_l_average*10**9)
print('calculated l[nm] = ',average_l*10**9)

print('effective Diff coeff = ',effective_D_average)
print('calculated Ddiif coeff = ',calculated_D_average)

print('.......................................................')

#%%

plt.plot(np.arange(20,ldist,1),eff_dist/sum(eff_dist), linestyle = '-.' ) 
plt.axvline(x=effective_l_mode*10**9, linestyle = '-', color = 'b', label = 'mode')
plt.axvline(x=effective_l_median*10**9, linestyle = '-.' ,color = 'b', label = 'median')
plt.axvline(x=effective_l_average*10**9, linestyle = ':' , color = 'b', label = 'mean')
plt.legend()

plt.plot(np.arange(20,ldist,1),dist ) 
plt.axvline(x=mode_l*10**9, linestyle = '-',color = 'r')
plt.axvline(x=median_l*10**9, linestyle = '-.',color = 'r')
plt.axvline(x=average_l*10**9, linestyle = ':',color = 'r')
plt.legend()
plt.xlim(0,800)
plt.xlabel('length [nm]')
plt.show()


# %%
#plot D dist
lengths = np.arange(100,800,1)
D = rot_diff_par(lengths*10**(-9),ar(lengths))
D_max = rot_diff_par(lengths*10**(-9),ar_max(lengths))
D_min = rot_diff_par(lengths*10**(-9),ar_min(lengths))
D_agg = rot_diff_par(lengths*10**(-9),10)
#D_ar30 = rot_diff_par(lengths*10**(-9),30)
plt.plot(lengths,D)
plt.plot(lengths,D_max)
plt.plot(lengths,D_min)
plt.plot(lengths,D_agg, label = 'AR = 10')
#plt.plot(lengths,D_ar30, label = 'AR 30')
plt.axvline(x=mode_l*10**9, color = 'r', label = 'mode')
plt.axvline(x=median_l*10**9, color = 'b', label = 'median')
plt.axvline(x=average_l*10**9, color = 'g', label = 'mean')
plt.axvline(x=effective_l_mode*10**9, linestyle = '--', color = 'r', label = 'mode')
plt.axvline(x=effective_l_median*10**9, linestyle = '--', color = 'b', label = 'median')
plt.axvline(x=effective_l_average*10**9, linestyle = '--', color = 'g', label = 'mean')
plt.axhline(2, color = 'gray' , alpha = 0.5)
plt.legend()
#plt.ylim(0,200)
plt.xlabel('length [nm]')
plt.ylabel(r'Rotational diffusion parameter [$s^{-1}$]')
plt.xlim(100,500)
plt.grid('minor')
plt.show()

print('*******************************************************')
# %% CAN WE COMPUTE THE CURVE FROM THE RELAXATION?


t = np.linspace(0,3.5, 1000)
lengths = np.linspace(20,500,500)
eff_dist =  correction(lengths, *popt)

D_l = rot_diff_par(lengths*10**(-9),ar(lengths))

# effecctive relax
exps = []
for i in range(len(D_l)):
    exps.append(np.exp(-6*D_l[i]*t) )
w_exps = []
for i in range(len(eff_dist)) :
    w_exps.append(eff_dist[i]*exps[i] )

relax = []
somma_exps = 0
for j in range(len(t)):
    for i in range(len(D_l)):
        somma_exps = somma_exps  + w_exps[i][j]
    relax.append(somma_exps)
    somma_exps = 0

relax = relax/relax[0]

# relax non correction
relax_nc = []
somma_exps_nc = 0
for j in range(len(t)):
    for i in range(len(D_l)):
        somma_exps_nc = somma_exps_nc  + exps[i][j]
    relax_nc.append(somma_exps_nc)
    somma_exps_nc = 0

relax_nc = relax_nc/relax_nc[0]

plt.plot(t,relax, linestyle = '-.', label = 'effective distribution')
plt.plot(t,relax_nc, linestyle = ':', label = 'only number distribution')
#plt.plot(t,relax )
plt.grid('both')
plt.plot(t, np.exp(-(6*1.99*t)**0.54), label = 'from EC')
plt.xlim(0,1.75)
plt.legend()
plt.xlabel('time[s]')
plt.grid('both')
plt.show()




# %%
