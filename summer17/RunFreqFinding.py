# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:23:49 2016

@author: Conner
"""

import numpy as np
import main
import os, errno

####Required parameters to run
mass1 = float(raw_input('Mass of Neutron Star 1 is: '))
mass2 = float(raw_input('Mass of Neutron Star 2 is: '))
lambda1 = float(raw_input('$\Lambda$ of Neutron Star 1 is: '))
lambda2 = float(raw_input('$\Lambda$ of Neutron Star 2 is: '))
#mass1 = 1.2
#mass2 = 1.5
#lambda1 = 1372
#lambda2 = 359

total_mass = mass1+mass2
####Optional parameters to change
freq = np.arange(1000,3000,5)
cores = 4 #Number of cores you want to run this program on.
MatchStyle = 1 #1 for ending too early or late, 0 for no consideration
CutoffFreq = 680.15728588321906
CompleteMerger = 0 #1 for complete merger, 0 for max amplitude
Informationtosave = 0 #Leave as 0 for this script.
NumWave = 'data/HotokezakaData/01_ALF_120150_60.dat'
n_t, num_hp, num_hc, num_freq = \
        np.loadtxt(NumWave, dtype=float, usecols=[0,1,2,3],unpack=True) #load in numerical data
num_freq = num_freq*(1/2./np.pi/total_mass/(4.92686088e-6))
t_conversion = total_mass*(4.92686088e-6)
n_t = n_t*t_conversion #Convert into seconds. Particular to Hotokezaka data


if __name__ == '__main__':
    Errors = main.FreqFinder(mass1, mass2, lambda1, lambda2, n_t, num_hp, num_hc, \
    num_freq, freq, cores, CutoffFreq, CompleteMerger, MatchStyle,Informationtosave)
    Errors = np.transpose(Errors) # pairs (A_corr, phi_corr)_i into (A0, A1...), (r0, r1...)
    H = Errors[0] # first element of the array A's
    bestf = freq[np.argmax(H)] # frequency for largest Amplitude correlation
    print type(bestf)
    print 'The best frequency for the given parameters is ' + str(bestf)
    # np.savetxt('BestFrequency.txt', str(bestf))
    folderName = 'data' + '_' + str(mass1) + '_' + str(mass2) + '_' + \
    str(int(lambda1)) + '_' + str(int(lambda2))
    with open(folderName + '/BestFrequency.txt', 'w') as f:
        print >>f, bestf
