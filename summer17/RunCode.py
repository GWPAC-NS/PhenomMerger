# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:30:09 2016

@author: Conner
"""

import main
import numpy as np
import glob
import os, errno
import hybrid
import matplotlib.pyplot as plt

####Required parameters to run
mass1 = float(raw_input('Mass of Neutron Star 1 is: '))
mass2 = float(raw_input('Mass of Neutron Star 2 is: ' ))
lambda1 = float(raw_input('$\Lambda$ of Neutron Star 1 is: '))
lambda2 = float(raw_input('$\Lambda$ of Neutron Star 2 is: '))
#mass1 = 1.2
#mass2 = 1.5
#lambda1 = 1372
#lambda2 = 359
total_mass = mass1 + mass2
total_lambda = lambda1+ lambda2

folderName = 'data' + '_' + str(mass1) + '_' + str(mass2) + '_' + \
str(int(lambda1)) + '_' + str(int(lambda2))

try:
    os.mkdir(folderName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

#Optional Parameters to change code
freq = (2.85E6/total_lambda + -7.25E8/(total_lambda**2)+2.48E3)/2.7 #Phenomenological
MatchStyle = 1 #1 for ending too early or late, 0 for no consideration
CutoffFreq = 680.15728588321906
CompleteMerger = 0 #1 for complete merger, 0 for max amplitude
Informationtosave = 2 #0 for print statements, 1 for plots, 2 for text and plots
###Read in Numerical Data. This will change depending on the source
NumWave = 'data/HotokezakaData/01_ALF_120150_60.dat'
n_t, num_hp, num_hc, num_freq = \
        np.loadtxt(NumWave, dtype=float, usecols=[0,1,2,3],unpack=True) #load in numerical data
num_freq = num_freq*(1/2./np.pi/total_mass/(4.92686088e-6)) #convert to Hz
t_conversion = total_mass*(4.92686088e-6)
n_t = n_t*t_conversion #Convert into seconds. Particular to Hotokezaka data


if __name__ == '__main__':

    time, h_PN, phi_PN = main.PN_Wave(mass1,mass2,lambda1,lambda2,freq,freq)

    if Informationtosave == 2:
        print '===> Saving Waveform to Text'
        np.savetxt(str(folderName) + '/Waveform.txt',\
            np.transpose([time, h_PN, phi_PN]),delimiter = ' ')
    if Informationtosave >= 1:
        print '===> Graphing Waveform'
        plt.subplot(2,1,1)
        plt.title('Post-Newtonian h_22 GW Strain Match')
        plt.plot(time,np.real(h_PN),'b')
        plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
        plt.xlabel('t (s)')
        plt.ylabel('$h_plus$')
        plt.subplot(2,1,2)
        plt.plot(time,np.imag(h_PN),'b')
        plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
        plt.xlabel('t (s)')
        plt.ylabel('$h_cross_{GW}$')
        plt.savefig(str(folderName) + '/Post-Newtonian.png')
        plt.close()
    HybridMatch, RMS = hybrid.Match(mass1,mass2,n_t,num_hp,num_hc,num_freq,h_PN, time,
                                    CutoffFreq, CompleteMerger, MatchStyle, Informationtosave, folderName)
