# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:32:46 2016

@author: Conner
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import equations as eqs
import EOBhybrid as hyb
import scipy.io as spio

m1 = [1.2,1.3,1.3,1.35,1.4,1.2,1.2,1.3,1.3,1.3,1.35,1.2,1.2,1.25,1.25,1.3,1.3,1.3,1.35,1.4,1.45,1.2,1.25,1.3,1.3,1.3,1.35,1.4,1.45,1.2,1.3,1.3,1.35]
m2 = [1.5,1.3,1.4,1.35,1.4,1.4,1.5,1.3,1.4,1.5,1.35,1.4,1.5,1.35,1.45,1.3,1.4,1.5,1.35,1.4,1.45,1.5,1.45,1.3,1.4,1.6,1.35,1.4,1.45,1.5,1.3,1.4,1.35]
lambda1 = [1372.08, 883.517, 883.517, 710.414, 571.409, 630.334, 630.334, 392.303, 392.303, 392.303, 311.325, 2130.99, 2130.99, 1689.74, 1689.74, 1344.68, 1344.68, 1344.68, 1072.32, 855.928, 683.099, 3153.48, 2544.99, 2065.37, 2065.37, 2065.37, 1684.58, 1380.25, 1135.53, 785.619, 479.366, 479.366, 376.219]
lambda2 = [368.857, 883.517, 571.409, 710.414, 571.409, 247.809, 158.003, 392.303, 247.809, 158.003, 311.325, 855.928, 544.501, 1072.32, 683.099, 1344.68, 855.928, 544.501, 1072.32, 855.928, 683.099, 937.652, 1135.53, 2065.37, 1380.25, 645.455, 1684.58, 1380.25, 1135.53, 183.546, 479.366, 295.881, 376.219]
G = 6.67408E-11 #m^3/kg/s^2
c = 299792458 #m/s
solarmass_mpc = 2.0896826e19
mpc_meters = 3.08567758E22

#PhenomMatch = np.loadtxt('Resolution Analysis.txt')
#EOB = sorted(glob.glob('C:\Users\Conner\Dropbox\Coner\EOB*'), key = os.path.getmtime)
NumWave = sorted(glob.glob('data/HotokezakaData/*.dat'))
#EOB = np.loadtxt('D:\Anaconda\Projects\SummerThings\EOB\EOB_1602_5PNlogTidalNospin\tmpworking\WavInspl.mat')
H_EOB = [0]*33
R_EOB = [0]*33
H_Phenom = [0]*33
R_Phenom = [0]*33
H_TT4f = [0]*33
R_TT4f = [0]*33
H_TT4 = [0]*33
R_TT4 = [0]*33
TL = [0]*33
freqdata = sorted(glob.glob('data/allWaves/*/Bestfreq.txt'))

for i in xrange(33):
    ## there are folders missing, the following handles the exceptions
    try:
        EOBWave = spio.loadmat('data/EOBdata/tmpworking'\
        + str(i+1)+'/WavInspl.mat', struct_as_record=False, squeeze_me=True)
        EOBWave = EOBWave['wav']
    except Exception:
        pass

    mass1 = m1[i]
    mass2 = m2[i]
    totalMass = mass1+mass2
    l1 = lambda1[i]
    l2 = lambda2[i]
    lambdaTotal = l1 + l2
    bestFreq= np.loadtxt(freqdata[i])
    fmode1 = eqs.fmodeCalc(mass1, l1)
    fmode2 = eqs.fmodeCalc(mass2, l2)

    Phenom_t, Phenom_fGW, Phenom_h, Phenom_phi = \
    eqs.PN_wave(m1[i], m2[i], lambda1[i], lambda2[i], bestFreq, bestFreq)
    EOBWave_t = EOBWave.t
    EOBWave_h = np.conj(EOBWave.psilm[1])

    TT4f_t, TT4f_fGW, TT4f_h, TT4f_phi = \
    eqs.PN_wave(m1[i], m2[i], lambda1[i], lambda2[i], fmode1,fmode2)
    TT4_t, TT4_fGW, TT4_h, TT4_phi = \
    eqs.PN_wave(m1[i], m2[i], lambda1[i], lambda2[i], 9999999, 9999999)
    conversion = totalMass * (4.92686088e-6)
    NumWave_t, NumWave_hp, NumWave_hc, freq = \
    np.loadtxt(NumWave[i], usecols = [0,1,2,3], unpack = True)
    freq = freq*(1/2./np.pi/totalMass/(4.92686088e-6))
#Need the Amplitudes to match the Numerical
    #EOBWave_h  = (EOBWave_hp + EOBWave_hc*1j)/totalMass*solarmass_mpc*0.63
#    EOBWave_hp = (EOBWave_h.real)*3.08567758E22

    ###Check the plots to match amplitudes and times
#    plt.plot(EOBWave_t, EOBWave_h.real, NumWave_t*conversion, NumWave_hp)
    #plt.plot(Phenom_t, Phenom_h.real, NumWave_t*conversion, NumWave_hp)
    #plt.show(1)

#    if os.path.exists(str(float(i))):
#        print('\n===ERROR=== folder already exists, \
#            use a different name please!\n')
#        break
#    else:
#        os.mkdir(str(float(i)))
#        (m_1,m_2,num_t,num_hp,num_hc,freq,PN_h_wave, PN_tc, cutoff, merger, style, save,i)

    H_EOB[i], R_EOB[i] = hyb.Match(mass1,mass2,NumWave_t,NumWave_hp, \
    NumWave_hc,freq,EOBWave_h*0.63,EOBWave_t*conversion, 680, 1,1,0,i)
    H_Phenom[i], R_Phenom[i] = hyb.Match(mass1,mass2,NumWave_t,NumWave_hp,\
    NumWave_hc,freq,Phenom_h,Phenom_t, 680, 1,1,0,i)

    H_TT4f[i], R_TT4f[i] = hyb.Match(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
    ,freq,TT4f_h,TT4f_t, 680, 1,1,0,i)
    H_TT4[i], R_TT4[i] = hyb.Match(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
    ,freq,TT4_h,TT4_t, 680, 1,1,0,i)
    TL[i] =lambdaTotal
    print '\n\n\n-------- ' + str(i) + ' HAS BEEN COMPLETED --------\n\n\n'

#for i in xrange(22,31):
#    EOBWave = spio.loadmat('D:\\anaconda\\Projects\\SummerThings\\EOB\\EOBstuff\\tmpworking'\
#    + str(i+1)+'\\WavInspl.mat', struct_as_record=False, squeeze_me=True)
#    EOBWave = EOBWave['wav']
#    mass1 = m1[i]
#    mass2 = m2[i]
#    totalMass = mass1+mass2
#    l1 = lambda1[i]
#    l2 = lambda2[i]
#    lambdaTotal = l1 + l2
#    fcalc= (2.85E6/lambdaTotal + -7.25E8/(lambdaTotal**2)+2.48E3)/2.7
#    Phenom_h, Phenom_t = main.main(m1[i],m2[i],lambda1[i],lambda2[i],fcalc,fcalc)
#    TT4f_h, TT4f_t = main.main(m1[i],m2[i],lambda1[i],lambda2[i],fmode1,fmode2)
#    TT4_h, TT4_t = main.main(m1[i],m2[i],lambda1[i],lambda2[i],99999,99999)
#    conversion = totalMass*(4.92686088e-6)
#    EOBWave_t = EOBWave.t
#    EOBWave_h = np.conj(EOBWave.psilm[1])
#    NumWave_t, NumWave_hp, NumWave_hc, freq = \
#    np.loadtxt(NumWave[i], usecols = [0,1,2,3], unpack = True)
#    freq = freq*(1/2./np.pi/totalMass/(4.92686088e-6))
#
##Need the Amplitudes to match the Numerical
#    #EOBWave_h  = (EOBWave_hp + EOBWave_hc*1j)/totalMass*solarmass_mpc*0.63
##    EOBWave_hp = (EOBWave_h.real)*3.08567758E22
#
#    ###Check the plots to match amplitudes and times
##    plt.plot(EOBWave_t, EOBWave_h.real, NumWave_t*conversion, NumWave_hp)
#    #plt.plot(Phenom_t, Phenom_h.real, NumWave_t*conversion, NumWave_hp)
#    #plt.show(1)
#
##    if os.path.exists(str(float(i))):
##        print('\n===ERROR=== folder already exists, \
##            use a different name please!\n')
##        break
##    else:
##        os.mkdir(str(float(i)))
#
#    H_EOB[i], R_EOB[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,EOBWave_h*0.63,EOBWave_t*conversion)
#    H_Phenom[i], R_Phenom[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,Phenom_h,Phenom_t)
#    H_TT4f[i], R_TT4f[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,TT4f_h,TT4f_t)
#    H_TT4[i], R_TT4[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,TT4_h,TT4_t)
#    TL[i] =lambdaTotal
#    print str(i)+' has been completed'
#
#for i in xrange (32,33):
#    EOBWave = spio.loadmat('D:\\anaconda\\Projects\\SummerThings\\EOB\\EOBstuff\\tmpworking'\
#    + str(i+1)+'\\WavInspl.mat', struct_as_record=False, squeeze_me=True)
#    EOBWave = EOBWave['wav']
#    mass1 = m1[i]
#    mass2 = m2[i]
#    totalMass = mass1+mass2
#    l1 = lambda1[i]
#    l2 = lambda2[i]
#    lambdaTotal = l1 + l2
#    fcalc= (2.85E6/lambdaTotal + -7.25E8/(lambdaTotal**2)+2.48E3)/2.7
#    Phenom_h, Phenom_t = main.main(m1[i],m2[i],lambda1[i],lambda2[i],fcalc,fcalc)
#    TT4f_h, TT4f_t = main.main(m1[i],m2[i],lambda1[i],lambda2[i],fmode1,fmode2)
#    TT4_h, TT4_t = main.main(m1[i],m2[i],lambda1[i],lambda2[i],99999,99999)
#    conversion = totalMass*(4.92686088e-6)
#    EOBWave_t = EOBWave.t
#    EOBWave_h = np.conj(EOBWave.psilm[1])
#    NumWave_t, NumWave_hp, NumWave_hc, freq = \
#    np.loadtxt(NumWave[i], usecols = [0,1,2,3], unpack = True)
#    freq = freq*(1/2./np.pi/totalMass/(4.92686088e-6))

#Need the Amplitudes to match the Numerical
    #EOBWave_h  = (EOBWave_hp + EOBWave_hc*1j)/totalMass*solarmass_mpc*0.63
#    EOBWave_hp = (EOBWave_h.real)*3.08567758E22

    ###Check the plots to match amplitudes and times
#    plt.plot(EOBWave_t, EOBWave_h.real, NumWave_t*conversion, NumWave_hp)
    #plt.plot(Phenom_t, Phenom_h.real, NumWave_t*conversion, NumWave_hp)
    #plt.show(1)

#    if os.path.exists(str(float(i))):
#        print('\n===ERROR=== folder already exists, \
#            use a different name please!\n')
#        break
#    else:
#        os.mkdir(str(float(i)))

#    H_EOB[i], R_EOB[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,EOBWave_h*0.63,EOBWave_t*conversion)
#    H_Phenom[i], R_Phenom[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,Phenom_h,Phenom_t)
#    H_TT4f[i], R_TT4f[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,TT4f_h,TT4f_t)
#    H_TT4[i], R_TT4[i] = hybrid.hybrid(mass1,mass2,NumWave_t,NumWave_hp,NumWave_hc\
#    ,freq,TT4_h,TT4_t)
#    TL[i] =lambdaTotal
#    print str(i)+' has been completed'

#TL = np.delete(TL,[21,31],0)
#H_EOB = np.delete(H_EOB,[21,31],0)
#H_Phenom = np.delete(H_Phenom,[21,31],0)
#H_TT4f = np.delete(H_TT4f,[21,31],0)
#H_TT4 = np.delete(H_TT4,[21,31],0)
#R_EOB = np.delete(R_EOB,[21,31],0)
#R_Phenom = np.delete(R_Phenom,[21,31],0)
#R_TT4f = np.delete(R_TT4f,[21,31],0)
#R_TT4 = np.delete(R_TT4,[21,31],0)
#plt.title('Relationship between Two Different PN Methods')
#np.savetxt('errorsPhenom.txt',np.transpose([H_EOB, H_Phenom, R_EOB, R_Phenom]),\
# delimiter = ' ')

plt.plot(TL_EOB, H_EOB_clean,'o')
plt.show()
plt.close()
# plt.plot(TL, H_Phenom,'*', TL, H_TT4f, 'D', TL, H_TT4, '+')
# plt.xlabel('Total Deformability')
# plt.ylabel('Hybrid Match')
# plt.legend(('BestFreq','f-mode TT4','Tidal T4'), loc = 3, prop ={'size':12})
# plt.show()
# plt.close()
