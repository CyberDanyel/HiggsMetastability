#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:02:42 2021

@author: samueltsang
"""

import numpy as np
import scipy.special as sps
import scipy.integrate as spi
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

m = 1.8e13 #GeV, mass of inflaton
M_P = 2.435e18 #GeV, reduced Planck mass
lamb = 0.01 #Higgs self-interaction parameter
lamb1 = 1.4e-13 #Potential parameter in quartic chaotic inflation

#Quadratic chaotic inflation model
comove = 60 + np.log(np.sqrt(m*M_P)/1e16) #Comoving Hubble length
a = 3.21*np.exp(comove) #Useful values
b = np.sqrt(np.pi/2/np.e) #Useful values
c = b*sps.erfi(1/np.sqrt(2)) #Useful values
N_max = M_P**2/2/m**2 #Maximum number of e-foldings for the Higgs potential not to exceed Planck scale energies
def f(N): #Define the integrand
    return (np.exp(-N)*np.sqrt(1+2*N)*(a-c+b*sps.erfi(np.sqrt(N+1/2))))**3
integral = spi.quad(f,0,60)[0] #Perform numerical integration from 0 to 60 e-foldings
def nmc(N_start): #Return the non-minimal coupling parameter given N_start with any value in [0, N_max]
    if N_start < 60:
        func = spi.quad(f,0,N_start)[0]
        return np.sqrt(lamb/6*np.log(4*np.pi/3*func))/4/np.pi
    else:
        func = integral + N_start - 60
        return np.sqrt(lamb/6*np.log(4*np.pi/3*func))/4/np.pi
def nmc_array(N_array): #Return a numpy array of non-minimal coupling parameter for an array of N_start
    values = []
    for i in N_array:
        values.append(nmc(i))
    return np.array(values)
nmc_value = nmc(60) #Find the non-minimal coupling parameter here
print('The non-minmal coupling parameter from quadratic chaotic inflation is %f.' %nmc_value)

#Quartic chaotic inflation model
comove1 = 60 + np.log(2*M_P*lamb1**(1/4)/1e16)
a1 = 3.21*np.exp(comove+1)
b1 = sps.expi(1)
N_max1 = 1/(4*np.sqrt(lamb1))
def f1(N):
    return (N*np.exp(-N)*(a1+sps.expi(N)-b1))**3
integral1 = spi.quad(f1,1,100)[0] #Just for use in the function later
def nmc1(N_start):
    if N_start < 100:
        func = spi.quad(f1,1,N_start+1)[0]
        return np.sqrt(lamb/6*np.log(4*np.pi/3*func))/4/np.pi
    else:
        func = integral1 + N_start - 100
        return np.sqrt(lamb/6*np.log(4*np.pi/3*func))/4/np.pi
def nmc_array1(N_array): #Return a numpy array of non-minimal coupling parameter for an array of N_start
    values = []
    for i in N_array:
        values.append(nmc1(i))
    return np.array(values)
nmc_value1 = nmc1(60)
print('The non-minmal coupling parameter from quartic chaotic inflation is %f.' %nmc_value1)
deviation = (nmc_value1 - nmc_value) / (nmc_value + nmc_value1) * 2 * 100 #percentage
print('The percentage deviation between quadratic and quartic chaotic inflation is %f%%.' %deviation)

#%%
fig, ax = plt.subplots() #Just a simple plot with tons of fancy settings
N_max = np.round(N_max)
N_range = np.logspace(0,9)
ax.plot(N_range,nmc_array(N_range)*1e3,color='black',label=r'$\langle \mathcal{N} \rangle = 1$ ($\phi^2$)') #Note normalisation
ax.plot(N_range,nmc_array1(N_range)*1e3,color='red',label=r'$\langle \mathcal{N} \rangle = 1$ ($\phi^4$)') #Note normalisation
ax.fill_between(N_range,nmc_array(N_range)*1e3,color='darkgrey',label=r'$\langle \mathcal{N} \rangle > 1$ ($\phi^2$)') #Note normalisation
ax.fill_between(N_range,nmc_array1(N_range)*1e3,nmc_array(N_range)*1e3,color='lightpink',label=r'$\langle \mathcal{N} \rangle > 1$ ($\phi^4$)') #Note normalisation
#ax.plot((N_max,N_max),(43.975,44.05),'--',color='black')
#ax.plot((N_max1,N_max1),(43.975,44.05),'--',color='red')
ax.set_xlabel(r'Number of e-foldings $N$')
ax.set_ylabel(r'Non-minimal coupling parameter $\xi$ ($\times 10^{-3}$)')
ax.set_xlim(left=4, right=1e9)
ax.set_ylim(bottom=43.975, top=44.05)
ax.set_xscale('log')
#ax.legend(frameon=False)
ax.tick_params(direction='in',which='both')
ax.text(0.79,0.35,'Planck scale',transform=ax.transAxes)
ax.text(0.40,0.13,r'$\langle \mathcal{N}_{\mathrm{quadratic}} \rangle > 1$',transform=ax.transAxes)
ax.text(0.42,0.47,r'$\langle \mathcal{N}_{\mathrm{quartic}} \rangle > 1$',transform=ax.transAxes)
ax.text(0.44,0.82,r'$\langle \mathcal{N} \rangle < 1$',transform=ax.transAxes)
ax.text(0.04,0.335,r'$\langle \mathcal{N}_{\mathrm{quadratic}} \rangle = 1$',transform=ax.transAxes)
ax.text(0.04,0.720,r'$\langle \mathcal{N}_{\mathrm{quartic}} \rangle = 1$',transform=ax.transAxes)
ax.arrow(0.80,0.33,0.15,0,head_width=0.008,linewidth=0.7,transform=ax.transAxes,color='black')
ax.minorticks_on()
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
#plt.savefig('non-minimal_coupling_v2.eps', format='eps') #Save your figure here
plt.show()

#We conclude that the non-minimal coupling is bounded below by 0.044.
#Note that our value(s) is lower than the paper published by Andreas and Arttu.
#They find that the lower bound for non-minimal coupling is 0.06 for both models.
#This is because they have quantum loop corrections, and do not assume slow-roll conditions.
