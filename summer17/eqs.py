# -*- coding: utf-8 -*-

import numpy as np
EulerGamma = np.euler_gamma
M_SUN = 4.92686088e-6  # Mass of the Sun [s]

def fmode(m,Lambda):
    """
    *Currently not being used*
    Calculate the resonant frequency, f-mode, using the tidal deformability
    coefficient lambda. This is the frequency found in the paper by Chan et. al.
    http://arxiv.org/pdf/1408.3789v2.pdf

    :param m: Mass of the neutron star in solar masses
    :type m: float
    :param Lambda: Unitless weighted deformability of the star
    :type Lambda: float

    """
    a0 = 1.820e-1
    a1 = -6.836e-3
    a2 = -4.196e-3
    a3 = 5.215e-4
    a4 = -1.857e-5
    x = np.log(Lambda/5.)
    omega = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    return (omega / ( 2* np.pi) / (m * M_SUN))

def t4_35PN(x,M,m1):
    """This is the PN t4 aproximation for a point particle. M is the total mass of the system.
    m1 and m2 are mass of the first and second star respectively. m2 is therefore redefined to
    be M-m1, if m1+m2=M. x is the post newtonian expansion parameter.
    m1,m2,m3, and x are of type: float

    :param x: Post-Newtonian expansion parameter
    :type x: float
    :param M: Total Mass of the System
    :type M: float
    :param m1: Mass of neutron star 1
    :type m1: float
    """
    m2=(M-m1)
    point =(1. + (-743./336. - \
    (11.*m1*m2)/(4.*np.power(m1 + m2,2)))*x + 4.*np.pi*np.power(x,1.5) + \
    ((34103. + (59472.*np.power(m1,2)*np.power(m2,2.))/np.power(m1 + m2,4.) + \
    (122949.*m1*m2)/np.power(m1 + m2,2))*np.power(x,2.))/18144. - ((4159. + \
    (15876.*m1*m2)/np.power(m1 + m2,2))*np.pi*np.power(x,2.5))/672. + \
    (5.*(-2649. + (146392.*np.power(m1,2.)*np.power(m2,2.))/np.power(m1 + \
    m2,4.) + (143470.*m1*m2)/np.power(m1 + \
    m2,2.))*np.pi*np.power(x,3.5))/12096. + \
    np.power(x,3.)*(16447322263./139708800. - (1712.*EulerGamma)/105. - \
    (5605.*np.power(m1,3.)*np.power(m2,3.))/(2592.*np.power(m1 + m2,6.)) + \
    (541.*np.power(m1,2.)*np.power(m2,2.))/(896.*np.power(m1 + m2,4.)) - \
    (56198689.*m1*m2)/(217728.*np.power(m1 + m2,2.)) + \
    (16.*np.power(np.pi,2))/3. + \
    (451.*m1*m2*np.power(np.pi,2))/(48.*np.power(m1 + m2,2)) - \
    (856.*np.log(16))/105. - \
    (1712.*np.log(np.sqrt(x)))/105.))
    return point

def LOPN(x,m,M):
    """
    The leading order Post-Newtonian

    :param x: Post-Newtonian expansion parameter
    :type x: float
    :param m: Mass of the neutron star
    :type m: float
    :param M: Total mass of the system
    :type M: float
    """
    lead = (64.*m*(M-m)*np.power(x,5.))/(5.*np.power(M,2.))
    return lead
def Tidal_Res(x,m,M,lam,x0):
    """
    The loss of energy due to tidal deformability

    :param x: Post-Newtonian expansion parameter
    :type x: float
    :param m: Mass of the neutron star
    :type m: float
    :param M: Total mass of the system
    :type M: float
    :param x0: The resonance frequency of the system/neutron star
    :type x0: float

    """
    #Revisited = -(3.*m**4.*x**5.*(8.*m*x0**3.*(x**3.-x0**3.)**2.+(M-m)*(-22.*x**9.+95.*x**6.*x0**3.\
    #-133.*x**3.*x0**6.+ 96.*x0**9.))*lam)/(4.*M**5.*(x**3.-x0**3.)**3.)

    TidalRes = (-3.*np.power(m,4.)*np.power(x,5.)*(8.*m*np.\
    power(x0,3.)*np.power(np.power(x,3.) - np.power(x0,3.),2.) + \
    (M-m)*(-22.*np.power(x,9.) + 95.*np.power(x,6.)*np.power(x0,3.) - \
    133.*np.power(x,3.)*np.power(x0,6.) + 96.*np.power(x0,9.)))*\
    (lam))/((4.*np.power(M,5.)*np.power(np.power(x,3.) - np.power(x0,3.),3.)))
    return TidalRes

def Read(x, M, m1, m2, lam1, lam2, x01,x02):
    """
    The full differential equation for the orbital motion of the compact
    binary neutron stars.

    :param m1: Mass of neutron star 1
    :type m1: float
    :param m2: Mass of neutron star 2
    :type m2: float
    :param lam1: The weighted deformability of neutron star 1
    :type lam1: float
    :param lam2: The weighted deformability of neutron star 1
    :type lam2: float
    :param x01: The resonance frequency of neutron star 1 in units of the post-newtonian parameter
    :type x01: float
    :param x02: The resonance frequency of neutron star 1 in units of the post-newtonian parameter
    :type x02: float
    """
    function = LOPN(x,m1,M)*(t4_35PN(x,M,m1) + Tidal_Res(x,m1,M,lam1,x01) + Tidal_Res(x,m2,M,lam2,x02))

    return function


def truncate(sol, t, M):

    """Calculate and truncate :math:`M\omega_{22}`, :math:`\phi` and
    :math:`t` when freq ~ 3000 Hz.

    :param sol: 2D array, with columns :math:`\phi` and :math:`x`,
        result from the integration of the phasing models
    :type sol: numpy.ndarray
    :param t: time step
    :type t: int
    :returns: :math:`M\omega_{22} = 2x^{3/2}`, frequency in Hz
        :math:`\phi`, phase in rad.
        :math:`t = M_{Sun} \cdot time\;step`, numpy.array, float

	"""
    Mw22 = np.array([2.*a**(1.5) for a in sol[:, 1] if
        ((2.*a**(1.5) <= 0.2) and a!=0.)])
    Mw22 = Mw22/(2. * np.pi * M_SUN * M)
    phi = sol[:len(Mw22), 0]
    t_new = M_SUN * t[:len(Mw22)]*M
    return Mw22, phi, t_new

def h22(x):
    """
    The amplitude of the waveform as a function of the post-Newtonian expansion parameter
    taken from Dr. Read's Mathematica code.

    :param x: Post-Newtonian expansion parameter
    :type x: float

    """
    ampl =(x*(1. - (373.*x)/168. + 2.*np.pi*np.power(x,1.5) - \
    (62653.*np.power(x,2))/24192. - (complex(0,6) + \
    (197.*np.pi)/42.)*np.power(x,2.5) + np.power(x,3.)*(43876092677./1117670400. \
    - (856.*EulerGamma)/105. + complex(0,428./105.)*np.pi + \
    (99.*np.power(np.pi,2))/128. - (1712*np.log(2))/105. - \
    (428.*np.log(x))/105.)))
    return ampl
