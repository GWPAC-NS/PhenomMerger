import numpy as np
from scipy.integrate import odeint

## -- CONSTANTS --
M_SUN = 4.92686088e-6  # Mass of the Sun [s]
C = 299792458  # speed of light [m/s]
EulerGamma = np.euler_gamma

## -- FUNCTIONS USED TO CALCULATE THE TERMS FOR THE DIFFERENTIAL EQUATION
def leadOrder_PN(x, m, M):
    lead = (64.*m*(M-m)*np.power(x,5.))/(5.*np.power(M,2.))
    return lead

def TaylorT4_PN(x, m1, M):
    m2=(M-m1)
    point = (1. + (-743./336. - (11.*m1*m2)/(4.*np.power(m1 + m2,2)))*x + \
    4.*np.pi*np.power(x,1.5) +
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

def tidalRes(x,m,M,lam,x0):
    tidalRes = (-3.*np.power(m,4.)*np.power(x,5.)*(8.*m*np.power(x0,3.)* \
    np.power(np.power(x,3.) - np.power(x0,3.),2.) + \
    (M-m)*(-22.*np.power(x,9.) + 95.*np.power(x,6.)*np.power(x0,3.) - \
    133.*np.power(x,3.)*np.power(x0,6.) + 96.*np.power(x0,9.)))*\
    (lam))/((4.*np.power(M,5.)*np.power(np.power(x,3.) - np.power(x0,3.),3.)))
    return tidalRes

def finalFunction(x, m1, m2, lam1, lam2, x01,x02):
    M = m1 + m2
    function = leadOrder_PN(x, m1, M) * (TaylorT4_PN(x, m1, M) + \
    tidalRes(x, m1, M, lam1, x01) + tidalRes(x, m2, M, lam2, x02))
    return function

def fmodeCalc(m, lam):
    a0 = 1.820e-1
    a1 = -6.836e-3
    a2 = -4.196e-3
    a3 = 5.215e-4
    a4 = -1.857e-5
    x = np.log(lam/5.)
    omega = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    return (omega / ( 2 * np.pi) / (m * M_SUN))

def h22(x):
    amplitude =(x*(1. - (373.*x)/168. + 2.*np.pi*np.power(x,1.5) - \
    (62653.*np.power(x,2))/24192. - (complex(0,6) + \
    (197.*np.pi)/42.)*np.power(x,2.5) + np.power(x,3.)*(43876092677./1117670400. \
    - (856.*EulerGamma)/105. + complex(0,428./105.)*np.pi + \
    (99.*np.power(np.pi,2))/128. - (1712*np.log(2))/105. - \
    (428.*np.log(x))/105.)))
    return amplitude

## -- INTEGRATION OF THE DIFFERENTIAL EQUATION
def PN_wave(m1, m2, lam1, lam2, f01, f02):
    totalM = m1 + m2
    # define PN expansion parameter x0
    x01 = (totalM * M_SUN * np.pi * f01)**(2./3.)
    x02 = (totalM * M_SUN * np.pi * f02)**(2./3.)
    # initial Conditions
    phi_0 = 0.
    x_0 = np.power(0.004,(2./3.))
    y_0 = (phi_0, x_0)
    # set up differential equations
    def toIntegrate(y, t):
        # dphi/dt = x^(3/2)
        # dx/dt = finalFunction()
        return [y[1]**(3./2.), finalFunction(y[1], m1, m2, lam1, lam2, x01, x02)]
    # integration of the differential equation
    t = np.arange(0, 300000)
    solution = odeint(toIntegrate, y_0, t)
    # solution[:, 0] = phi_PN
    # solution[:, 1] = x_PN.. but we have to stop before the function goes bananas

    # calculation of the waveform: amplitude, frequency, phase and time
    fGW_PN = truncate(solution, totalM)
    t_PN = M_SUN * t[:len(fGW_PN)] * totalM
    phi_PN = solution[:len(fGW_PN), 0]
    x_PN =  solution[:len(fGW_PN), 1]
    ampl_PN = h22(x_PN) * np.exp(2j*phi_PN)
    return t_PN, fGW_PN, ampl_PN, phi_PN
    # function returns time (s), frequency (Hz), Ampl (dim.less), phi (rad)


## -- RECOVERY OF THE FREQUENCY FROM THE SOLUTION, TRUNCATING AT SINGULARITY
def truncate(sol, M):
    newSol = [0]*300000
    for i in range(len(sol[:, 1])):
        newSol[i] = sol[i, 1]
        # keep only values that increase monotonically
        if sol[i, 1] < sol[i-1, 1]:
            break

    Mw22 = np.array([2.*a**(1.5) for a in newSol # dimensionless GW freq
    if ((2.*a**(1.5) <= 0.25) and a != 0.0)])
    f_GW = Mw22 / (2. * np.pi * M_SUN * M) # GW freq in Hz
    return f_GW
