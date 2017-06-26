import eqs
import numpy as np
import hybrid
from multiprocessing import Pool, freeze_support
from functools import partial
from scipy.integrate import odeint

#List of constants
M_SUN = 4.92686088e-6  # Mass of the Sun [s]
C = 299792458  # speed of light [m/s]

def FreqFinder(mass1, mass2, l1,l2,n_t,num_hp, num_hc, freq, f, cores, cutoff,\
merge,style,save):
    """
    Code that finds the frequency that yields the largest hybrid match value.
    This code takes advantage of multiple cores.

    :param mass1: Mass of neutron star 1
    :type mass1: float
    :param mass2: Mass of neutron star 2
    :type mass2: float
    :param l1: Lambda parameter of neutron star 1
    :type l1: float
    :param l2: Lambda parameter of neutron star 2
    :type l2: float
    :param n_t: The time sequence for the numerical waveform
    :type n_t: Array of floats
    :param num_hp: Plus polarization of the gravitational wave (real part)
    :type num_hp:Array of floats
    :param num_hc: Cross polarization of the gravitational wave (imaginary part)
    :type num_hc: Array of floats
    :param freq: Frequency from the numerical wave in Hz
    :type freq: Array of floats
    :param f: Frequency range to find the best match
    :type f: Array of floats
    :param cores: Number of cores you want to run this script with
    :type cores: Float
    :param cutoff: The beginning cutoff frequency for the numerical match region
    :type cutoff: Float
    :param merge: The type of merger information you want (max amp. or full merger)
    :type merge: Float
    :param style: The type of hybrid match you want to use (0 after match region or\
    no consideration)
    :type style: Float
    :param save: The amount of information you want to save (none, graphs, txt)
    :type save: Float

    """
    freeze_support()
    print '===> Initializing. Starting frequency finding for your parameters.'
    p = Pool(cores) #Number of cores to use
    func = partial(all_in_one, mass1,mass2,l1,l2,n_t, num_hp, num_hc,freq,\
    cutoff,merge,style,save) #All the parameters that stay constant
    H = p.map(func,f) #Run the program while iterating over f - f is the
    ## frequency range we declared (1000-3000, increments of 5)
    p.close()
    p.join()
    return H

def all_in_one(m1,m2,l1,l2,n_t, num_hp, num_hc,freq,cutoff,merge,style,save,f0):
    """
    Code necessary to run multiple core usage with a varying frequency.
    """
    inFolder = 'data' + str(m1) + '_' + str(m2) + '_' + str(l1) + '_' + \
    str(l2) + '_'
    PN_t, PN_h, PN_phi = PN_Wave(m1,m2,l1,l2,f0,f0)
    H = hybrid.Match(m1,m2,n_t,num_hp,num_hc,freq,PN_h, PN_t,cutoff,merge,style,save, inFolder)
    return H

def PN_Wave(mass_1,mass_2,Lambda1, Lambda2, f01,f02):
    """
    This function will generate a waveform that is consistent with Dr. Read's
    Mathematica code and is taken from Veronica's code ResTid.

    :return t_PN, a_PN, phi_PN: A 2-D array with the time in seconds, amplitude in NINJA units (?), \
    and the phase information of the wave (rads)
    """
    #Define an array of masses and lambda
    M = mass_1 + mass_2

    #Convert frequency (Hz) into PN expansion parameter
    x01 = (M*M_SUN*np.pi*f01)**(2./3.)
    x02 = (M*M_SUN*np.pi*f02)**(2./3.)

    #Intial Conditions
    phi_0 = 0.
    x_0 = np.power(0.004,(2./3.))
    y_0 = (phi_0, x_0)

    def f_Read(y,t):
        """Create an ordinary differential equation with the 3.5 PN approximation and the
        tidal corrections
        """
        return [y[1]**(3./2.),eqs.Read(y[1], M, mass_1, mass_2, Lambda1, Lambda2, x01, x02)]

    #Integrate the function
    t = np.arange(0,6000000)
    # int_PN_Read = odeint(f_Read, y_0, t)
    int_PN_Read = odeint(f_Read, y_0, t)

    #Pull out the frequency, angle, and time from the integration
    Mw22_PN, phi_PN, t_PN = \
    eqs.truncate(int_PN_Read, t, M)
    x_PN_Read = ((Mw22_PN * np.pi * M_SUN * M)**(2./3.))

    #Calculate the Amplitude of the Gravitational Wave
    a_PN = eqs.h22(x_PN_Read) * np.exp(2j*phi_PN)

    print '==> Waveform with ' + str(mass_1) +' '+ str(mass_2) +' ' + str(Lambda1) \
    + ' ' + str(Lambda2) + ' ' + str(f01) + ' ' + str(f02) +' has been completed'

    return t_PN, a_PN, phi_PN
