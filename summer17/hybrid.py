import scipy as sci
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import mean, sqrt, square

#######constants
G = 6.67384*(10^-11)
c = 299792458
#f_low = 70
#pi = np.pi
#sample_rate = 4096*10
#delta_t = 1.33e-5

def Match(m_1,m_2,num_t,num_hp,num_hc,freq,PN_h_wave, PN_tc, cutoff, merger, style, save, folder):
    """
    Hybrid match script that interpolates the waveforms, finds a match region, \
    matches the PN to the numerical, graphs the result, calculates the phase evolution,\
    and finds the RMS of the phase evolution

    :returns: Hybrid Match Value and RMS of the Phase Evolution
    """
    print '===> Calculating Hybrid and RMS Matches'

##### Define interpolation functions for Hotokezaka data, using the PN time scale

    delta_t = PN_tc[1]-PN_tc[0] #Gives us the time step of the calculated PN
    interpo_hp = sci.interpolate.interp1d(num_t,num_hp, kind = 'linear')
    interpo_hc = sci.interpolate.interp1d(num_t,num_hc, kind = 'linear')
    interpo_freq = sci.interpolate.interp1d(num_t,freq, kind = 'linear')

##### interpolate the numerical h_plus and h_cross with PN timeseries

    num_ts = np.arange(num_t[0],num_t[-1],delta_t)
    new_num_hp = interpo_hp(num_ts)
    new_num_hc = interpo_hc(num_ts)
    new_freq = interpo_freq(num_ts)

#Combine the plus and cross polarizations
    num_wave = new_num_hp + new_num_hc*1j

####Select match region
#Find the index to start: Time must be larger than 0.0005 and frequency higher than cutoff
    def Initial():
        n=0
        for line in num_ts:
            if num_ts[n]<0.0005 or new_freq[n]<cutoff:
                n+=1
            else:
                return n
    match_i = Initial() # index is the first one that fulfills conditions of Initial()

#### if merger = 1, keep going while current ampl is geater than next (until merger),
#### else keep going until the end
    if merger == 1:
        match_fold = np.argmax(np.sqrt(np.abs(num_wave)))
        n=0
        while np.sqrt(np.abs(num_wave[match_fold+n]))>np.sqrt(np.abs(num_wave[match_fold+n+1])):
            n+=1
        match_f = match_fold+n
    else:
        match_f = np.argmax(np.sqrt(np.abs(num_wave)))

############## Find a correlation between the PN and the numerical Hotokezaka waveforms
## style == 1 uses padding with zeros on either side of the waveform time interest interval
## else uses the whole waveform for the match

    if style == 1:
        print 'Using new method to calculate Hybrid and RMS match.'
        z = sig.correlate(PN_h_wave,num_wave[match_i:match_f],mode='full')
        # result z is a N-dimensional array of complex numbers
        abs_z = np.abs(z) # calculate the abs.val of each element of z
        phi = np.angle(z[np.argmax(abs_z)]) #calculate the angle of the element with largest abs.val

##### Next we find the max index number after subtracting the excess length given by full convolution which includes the len of the 2nd argument.

## w sets the time at which to start the match
        w = np.argmax(abs_z) - len(num_wave[match_i:match_f]) + 1
#    num_wave = np.concatenate((num_wave[match_i:match_f], np.zeros(len(num_wave) - match_f)),axis=0)
        num_tc = num_ts - num_ts[0] + (w - match_i)*delta_t

#### Quantifying Match Data using phase evolution and RMS
#Obtain Phase data from the waveforms (tangent inverse)
        phi_num = np.unwrap(np.angle(num_wave[:len(PN_tc[w:])]))
        phi_PN = np.unwrap(np.angle(PN_h_wave[w:]))
#Some corrections to the phase so that start from 0 to 2pi
        if phi > 0.:
            phi = phi - 2*np.pi
        phi_num_new = phi_num + phi
        if np.abs(phi_num_new[0]-phi_PN[0])>5:
            phi_PN = phi_PN - 2*np.pi
#### Calculate the root mean sqaured
        r =sqrt(mean(square(phi_num_new-phi_PN)))

#### Quantifying Match Data by Relating the Numerical to itself
#Add zeros to the end of the match region
        PN_h_wave = PN_h_wave[w:]
        num_wave = np.concatenate((num_wave[match_i:match_f], np.zeros(len(num_wave) - match_f)),axis=0)
        PN_h_wave = np.concatenate((PN_h_wave, np.zeros(len(num_wave)-len(PN_h_wave))),axis=0)

        numPN = sig.correlate(PN_h_wave,num_wave,mode='full') #Inner product of the PN and Numerical
        overlap = max(np.abs(numPN))
        self_corr = sig.correlate(num_wave,num_wave,mode='full') #Inner product of the Numerical with itself
        numnum = max(np.abs(self_corr))
        self_corPN = sig.correlate(PN_h_wave,PN_h_wave,mode='full') #Inner product of the PN with itself
        PNPN = max(np.abs(self_corPN))
        A = overlap/np.sqrt(numnum*PNPN) # Hybrid match value <num|PN>/sqrt(<num|num><PN|PN>)
        num_phase_shift = np.exp(1j*phi)*num_wave

#### Graph the Results of the Match
        if save >= 1:
#Graph the plus polarization of the wave
            plt.subplot(2,1,1)
            plt.title('GW Strain Match')
            plt.plot(num_tc[match_i:],np.real(PN_h_wave),'r',num_tc[match_i:],np.real(num_phase_shift), 'b')
            #plt.plot(num_tc[match_i:],np.real(num_phase_shift[match_i:]))
            plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
            plt.legend(('Post Newtonian', 'Numerical'), 'upper left', \
                fancybox = True)
            plt.xlabel('t (s)')
            plt.ylabel('$hplus$')
            plt.xlim(num_tc[match_i],PN_tc[-1])

#Graph the cross polarization of the wave
            plt.subplot(2,1,2)
            plt.plot(num_tc[match_i:],np.imag(PN_h_wave),'r',num_tc[match_i:],np.imag(num_phase_shift),'b')
    #plt.plot(num_tc[match_i:],np.imag(num_phase_shift[match_i:]))
            plt.xlim(num_tc[match_i],PN_tc[-1])
            plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
            plt.legend(('Post Newtonian', 'Numerical'), 'upper left', \
                fancybox = True)
            plt.xlabel('t (s)')
            plt.ylabel('$hcross $')
            plt.savefig(str(folder) + '/HybridMatch_zeros-padded.png')
            plt.close()

#### Graph the Phase Evolutions
            phi_num = phi_num_new - phi_num_new[0]
            phi_PN = phi_PN - phi_PN[0]
            plt.title('Phase vs. Time Comparison')
            plt.plot(PN_tc[w:],phi_num, 'b')
            plt.plot(PN_tc[w:],phi_PN, 'r')
            plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
            plt.legend(('Numerical Wave', 'Post Newtonian Wave'), 'upper left', \
                fancybox = True)
            plt.xlabel('t (s)')
            plt.ylabel('Phase (Radians)')
            plt.savefig(str(folder) + '/RMSGraph_zeros-padded.png')
            plt.close()

#### Save the information about the waveform into text files
        if save ==2:
            np.savetxt(str(folder) + '/Match_RMS_values_zeros-padded.txt', [A,r], delimiter = ' ')
        print '==> The Match for ' + str(m_1) + ' ' + str(m_2) + ' has been completed'
        print 'The Match is ' + str(A) + ' and the RMS is ' + str(r)
        return A, r

    else:
        print 'Using old method to calculate Hybrid and RMS Match.'
        z = sig.correlate(PN_h_wave,num_wave[match_i:match_f],mode='full')
        abs_z = np.abs(z)
        phi = np.angle(z[np.argmax(abs_z)])
        num_phase_shift = np.exp(1j*phi)*num_wave

##### Next we find the max index number after subtracting the excess length given by full convolution which includes the len of the 2nd argument.
        w = np.argmax(abs_z) - len(num_wave[match_i:match_f]) + 1
##### Time corrections
        num_tc = num_ts - num_ts[0] + (w - match_i)*delta_t

### Quantifying Match Data by Relating the Numerical to itself
        numPN = sig.correlate(PN_h_wave,num_wave[match_i:match_f],mode='full') #Inner product of the PN and Numerical
        overlap = max(np.abs(numPN))
#numnum = np.sum( num_wave[match_i:match_f] * np.conjugate(num_wave)[match_i:match_f] )
        self_corr = sig.correlate(num_wave[match_i:match_f],num_wave[match_i:match_f],mode='full') #Inner Product of the Numerical with itself
        numnum = max(np.abs(self_corr))
        self_corPN = sig.correlate(PN_h_wave[w:w+(match_f-match_i)],PN_h_wave[w:w+(match_f-match_i)],mode='full') #Inner Product of the PN with itself
        PNPN = max(np.abs(self_corPN))
        A = overlap/np.sqrt(numnum*PNPN) #Hybrid match value

#### Quantifying Match Data using phase evolution and RMS
#Obtain Phase data from the waveforms (tangent inverse)
        phi_num = np.unwrap(np.angle(num_wave[:len(PN_tc[w:])]))
        phi_PN = np.unwrap(np.angle(PN_h_wave[w:]))

#Some corrections to the phase so that start from 0 to 2pi
        if phi > 0.:
            phi = phi - 2*np.pi
        phi_num_new = phi_num + phi
        if np.abs(phi_num_new[0]-phi_PN[0])>1:
            phi_PN = phi_PN - 2*np.pi

####Calculate the RMS of the Phases
        r =sqrt(mean(square(phi_num_new-phi_PN)))

        if save >= 1:
### Graph the Plus polarization of the gravitational wave
            plt.subplot(2,1,1)
            plt.title('h_22 GW Strain Match')
            plt.plot(PN_tc,np.real(PN_h_wave),'r', num_tc[match_i:],np.real(num_phase_shift[match_i:]), 'b')
            plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
            plt.legend(('Post Newtonian', 'Numerical'), 'upper left', \
                fancybox = True)
            plt.xlabel('t (s)')
            plt.ylabel('$h_plus$')
            plt.xlim(num_tc[0],num_tc[match_f])

## Graph the Cross polarization of the gravitational wave
            plt.subplot(2,1,2)
            plt.plot(PN_tc,np.imag(PN_h_wave),'r',num_tc[match_i:],np.imag(num_phase_shift[match_i:]),'b')
            plt.xlim(num_tc[0],num_tc[match_f])
            plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
            plt.legend(('Post Newtonian', 'Numerical'), 'upper left', \
                fancybox = True)
            plt.xlabel('t (s)')
            plt.ylabel('$h_cross {GW}$')
            plt.savefig(str(folder) + '/HybridMatch.png')
            plt.xlim(num_tc[0],num_tc[match_f])
            plt.close()

####Graph the Phase evolution
            plt.title('Phase vs. Time Comparison')
            plt.plot(PN_tc[w:],phi_num, 'b')
            plt.plot(PN_tc[w:],phi_PN, 'r')
            plt.grid(b = True, which = 'major', color = 'k', linestyle = ':')
            plt.legend(('Numerical Wave', 'Post Newtonian Wave'), 'upper left', \
                fancybox = True)
            plt.xlabel('t (s)')
            plt.ylabel('Phase (Radians)')
            plt.savefig(str(folder) + '/RMSGraph.png')
            plt.close()
        if save ==2:
            np.savetxt(str(folder) + '/Match_RMS_values.txt', [A,r], delimiter = ' ')
    print '==> The Match for ' + str(m_1) + ' ' + str(m_2) + ' has been completed'
    print 'The Match is ' + str(A) + ' and the RMS is ' + str(r)
    return A, r
