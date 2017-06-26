
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats
import os

lambda1 = [1372.08, 883.517, 883.517, 710.414, 571.409, 630.334, 630.334, 392.303, 392.303, 392.303, 311.325, 2130.99, 2130.99, 1689.74, 1689.74, 1344.68, 1344.68, 1344.68, 1072.32, 855.928, 683.099, 3153.48, 2544.99, 2065.37, 2065.37, 2065.37, 1684.58, 1380.25, 1135.53, 785.619, 479.366, 479.366, 376.219]
lambda2 = [368.857, 883.517, 571.409, 710.414, 571.409, 247.809, 158.003, 392.303, 247.809, 158.003, 311.325, 855.928, 544.501, 1072.32, 683.099, 1344.68, 855.928, 544.501, 1072.32, 855.928, 683.099, 937.652, 1135.53, 2065.37, 1380.25, 645.455, 1684.58, 1380.25, 1135.53, 183.546, 479.366, 295.881, 376.219]
m1 = [1.2,1.3,1.3,1.35,1.4,1.2,1.2,1.3,1.3,1.3,1.35,1.2,1.2,1.25,1.25,1.3,1.3,1.3,1.35,1.4,1.45,1.2,1.25,1.3,1.3,1.3,1.35,1.4,1.45,1.2,1.3,1.3,1.35]
m2 = [1.5,1.3,1.4,1.35,1.4,1.4,1.5,1.3,1.4,1.5,1.35,1.4,1.5,1.35,1.45,1.3,1.4,1.5,1.35,1.4,1.45,1.5,1.45,1.3,1.4,1.6,1.35,1.4,1.45,1.5,1.3,1.4,1.35]
# lambda1, lambda2, m2 and m2 have 33 values each

f = xrange(1000,3000,5)
total_lambda= []
bestfreq = []
compare =[]
tmass = []
mdiff =[]
besterror = []
earlyfreq = []
hybriddiff=[]
laterfreq =[]

#### Load the data, os.path.getmtime makes sure the latest version is used
EarlyData = sorted(glob.glob('SummerThings/NewThings/*'))
LaterData = sorted(glob.glob('SummerThings/AdditionFreq/*'))
NTHybrid = np.loadtxt('ResonantModel/NEWNTData.txt', usecols=[0], dtype=float,\
delimiter = ' ')
#NTRMS = np.loadtxt('D:\Anaconda\Projects\ResonantModel\NEWNTData.txt', usecols=[1], dtype=float,delimiter = ' ')
NRHybrid = np.loadtxt('ResonantModel/NEWNRData.txt',usecols=[0], dtype=float, \
delimiter = ' ')
#radius1 = [0]*33
#radius2 = [0]*33
#masscheck1 = [0]*33
#masscheck2 = [0]*33
#
#for i in xrange(0,5,1):
#    mass1 = m1[i]
#    mass2 = m2[i]
#    radius1[i],masscheck1[i] = TOVOnly.radius(mass1,13.6624,4.070,2.411,1.890)
#    radius2[i],masscheck2[i] = TOVOnly.radius(mass2,13.6624,4.070,2.411,1.890)
#for i in xrange(5,11,1):
#    mass1 = m1[i]
#    mass2 = m2[i]
#    radius1[i],masscheck1[i] = TOVOnly.radius(mass1,13.3154,2.830,3.445,3.348)
#    radius2[i],masscheck2[i] = TOVOnly.radius(mass2,13.3154,2.830,3.445,3.348)
#for i in xrange(11,21,1):
#    mass1 = m1[i]
#    mass2 = m2[i]
#    radius1[i],masscheck1[i] = TOVOnly.radius(mass1,13.8154,2.909,2.246,2.144)
#    radius2[i],masscheck2[i] = TOVOnly.radius(mass2,13.8154,2.909,2.246,2.144)
#for i in xrange(21,29,1):
#    mass1 = m1[i]
#    mass2 = m2[i]
#    radius1[i],masscheck1[i] = TOVOnly.radius(mass1,13.634,2.224,3.033,1.325)
#    radius2[i],masscheck2[i] = TOVOnly.radius(mass2,13.634,2.224,3.033,1.325)
#for i in xrange(29,33,1):
#    mass1 = m1[i]
#    mass2 = m2[i]
#    radius1[i],masscheck1[i] = TOVOnly.radius(mass1,13.4304,3.005,2.988,2.851)
#    radius2[i],masscheck2[i] = TOVOnly.radius(mass2,13.4304,3.005,2.988,2.851)

#print radius1,radius2,masscheck1,masscheck2
Error = sorted(glob.glob('TrendFreqFinder/Data/*'))
pickem = [2,3,20]

for i in xrange(33):
    mass1 = m1[i]
    mass2 = m2[i]
#    rad1 = radius1[i]
#    rad2 = radius2[i]
    total_mass = mass1+mass2
    eta = (mass1*mass2)/np.power(total_mass,2)
    l1 = lambda1[i]
    l2 = lambda2[i]
#    radiusof1 = radius1[i]
#    radiusof2 = radius2[i]
    lamb = 8./13.*(((1+7*eta-31*np.power(eta,2))*(l1+l2))+\
    np.sqrt(1-4*eta)*(1+9*eta-11*np.power(eta,2))*(l1-l2))
    grapherrors = np.loadtxt(Error[i] + '/Errors.txt')
    earlydata = np.loadtxt(EarlyData[i] + '/Bestfreq.txt')
    laterdata = np.loadtxt(LaterData[i] + '/Freqdiff.txt')
    errorcompare = np.loadtxt(Error[i] + '/ErrorComparison.txt')
#    plt.plot(freq, grapherrors)
    graphfreq = np.loadtxt(Error[i] + '/Inputs.txt')
    freqbest= graphfreq[4] # input resonance frequencies (f same for BNS 1 & 2)
    earlierfreq = earlydata
#    plt.title('Tidal Deformability vs Resonance Frequency')
#    #plt.plot((l1+l2),freqbest*(mass1*mass2)**(3./5)/(mass1+mass2)**(1./5), 'o')
#    plt.plot((l1+l2),freqbest*total_mass, 'o')
#    plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
#    plt.xlabel('Lambda')
#    plt.ylabel('')
#    plt.show()
    tmass.append(mass1)
    earlyfreq.append(earlierfreq)
    laterfreq.append(laterdata[2])
    mdiff.append(mass2)
    compare.append(errorcompare[0])
    besterror.append(errorcompare[1])
    total_lambda.append(l1+l2)
    bestfreq.append(freqbest)
    hybriddiff.append(errorcompare[1]-NTHybrid[i])
#    plt.plot(lamb,freqbest)
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Hybrid Match')
#    plt.show()
##print np.size(total_lambda),np.size(bestfreq)
total_lambda = np.array(total_lambda)
bestfreq = np.array(bestfreq)
compare = np.array(compare)

def func(x,a,b,c):
    return a/x + b/(x**2)+c

ALF = xrange(0,5)
APR4 = xrange(5,11)
H4 = xrange(11,21)
MS1 = xrange(21,29)
SLy = xrange(29,33)
#
popt, cov = curve_fit(func,total_lambda,bestfreq)
perr = np.sqrt(np.diag(cov))
popt1, cov1 = curve_fit(func,total_lambda,earlyfreq)
popt2, cov2 = curve_fit(func,total_lambda,laterfreq)
#print popt
#print perr
x = np.linspace(min(total_lambda),max(total_lambda),100000)
p1 = popt[0]
p2 = popt[1]
p3 = popt[2]
#print p1, p2, p3
residuals = bestfreq - func(total_lambda,p1,p2,p3)
rsquared = sum(residuals**2.)
#print rsquared
#plt.plot(total_lambda[0:5],NTHybrid[0:5],'bo'\
#,total_lambda[5:11],NTHybrid[5:11],'go'\
#,total_lambda[11:21],NTHybrid[11:21],'ro'\
#,total_lambda[21:29],NTHybrid[21:29],'co'\
#,total_lambda[29:33],NTHybrid[29:33],'mo')
##plt.xlabel('Total Tidal Deformability')
##plt.ylabel('Best Frequency x Total Mass')
plt.plot(x,func(x,*popt),'b')
#plt.plot(x,func(x,*popt1),'g')
#plt.plot(x,func(x,*popt2),'r')
#plt.plot(total_lambda[0:5],NRHybrid[0:5],'b+'\
#,total_lambda[5:11],NRHybrid[5:11],'g+'\
#,total_lambda[11:21],NRHybrid[11:21],'r+'\
#,total_lambda[21:29],NRHybrid[21:29],'c+'\
#,total_lambda[29:33],NRHybrid[29:33],'m+')
#
#plt.plot(total_lambda[0:5],earlyfreq[0:5],'bD'\
#,total_lambda[5:11],earlyfreq[5:11],'gD'\
#,total_lambda[11:21],earlyfreq[11:21],'rD'\
#,total_lambda[21:29],earlyfreq[21:29],'cD'\
#,total_lambda[29:33],earlyfreq[29:33],'mD')
#
#plt.plot(total_lambda[0:5],bestfreq[0:5],'bo'\
#,total_lambda[5:11],bestfreq[5:11],'go'\
#,total_lambda[11:21],bestfreq[11:21],'ro'\
#,total_lambda[21:29],bestfreq[21:29],'co'\
#,total_lambda[29:33],bestfreq[29:33],'mo')
#
#
fig = plt.figure(1)
ax = fig.add_subplot(111)
line = plt.plot(total_lambda,bestfreq, 'bo', total_lambda,earlyfreq, 'gD', \
total_lambda,laterfreq, 'r*')

## The following adds arrows to identify each data point with its masses
# for i in np.arange(33):
#    ax.annotate(str(tmass[i])+' ' + str(mdiff[i]), xy=(total_lambda[i], laterfreq[i]),\
#        xycoords='data',\
#        xytext=(3*i,3*i), textcoords='offset points',\
#        arrowprops=dict(arrowstyle='->'))

##plt.title('Relationship Between Differing Match Region on Bestfit Frequency')
plt.ylabel('Frequency (Hz)', labelpad = 10)
plt.xlabel(u'Total $\Lambda$ = $\Lambda_1$ + $\Lambda_2$', labelpad = 15)
plt.legend(['CurveFitting','Old Match Region','Subtract Match Region','Add Match Region'],\
prop = {'size':12})
#plt.legend(['CurveFitting','Largest Match' ])
plt.show()
#print max(hybriddiff)
##
