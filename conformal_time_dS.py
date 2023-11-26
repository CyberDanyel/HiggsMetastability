#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 21:15:37 2021

@author: samueltsang
"""

#This is the code file of the summer research project titled 'Inflationary cosmology and the metastability of the Higgs vacuum'.

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

#Energy content of the universe
energy_dark = 0.69
energy_matter = 0.31
energy_radiation = 5.4e-5

#The Hubble constant at present H0 is
H0 = 1.5e-42 #GeV

#The scale factor at present a0 is
a0 = 1

#The differential conformal time dn is given by the function
def conformal_time(a, energy_dark, energy_matter, energy_radiation, a0):
    return 1/np.sqrt(energy_dark*a**4 + energy_matter*a0**3*a + energy_radiation*a0**4)#(/H0)

#The difference in conformal time dn = n_0 - n_inf is given by
args = (energy_dark, energy_matter, energy_radiation, a0)
dn = spi.quad(conformal_time, 0, a0, args=args)#/(a0*H0)
print(dn)

#We can visualise the conformal time function as follows
a = np.linspace(0,1,num=1001)
plt.plot(a, conformal_time(a, *args))
plt.show()
