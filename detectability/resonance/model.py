import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import odeint, simpson,quad
from scipy.interpolate import interp1d
from scipy.optimize import bisect, minimize_scalar, minimize

import os

from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.filter import overlap_cplx, matched_filter, sigma, sigmasq, match
from pycbc.psd.read import from_txt
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd.analytical import aLIGO140MpcT1800545, aLIGO175MpcT1800545, EinsteinTelescopeP1600143

from astropy import constants as const      # physical constants
from astropy import units as u              # astronomical units
c = const.c.value              # light speed (IS units, m/s)
G = const.G.value              # grav. const. (IS units, m³/(s²kg))
M_sun = u.solMass.to("kg")     # solar mass (kg)
msolsec = G*M_sun/c**3 # solar mass in seconds, natural units
jultosec = G/c**5 # from jules to seconds



class System():
    '''
    Physics describing the physical source
    '''
    def __init__(self, m1=1.4, m2=1.4):
        # mtotal and mchirp are converted to units of seconds from Msol
        self.mtotal = msolsec * (m1 + m2)
        self.eta = m1*m2/((m1 + m2)**2)
        self.mchirp = self.mtotal*(self.eta**(3./5))

    def v(self, f):
        '''
        Velocity as a function of frequency. 
        v = v(f)
        Units: [f] = Hz, [v] = -
        '''
        return (np.pi * self.mtotal * f)**(1./3)

    def dvdf(self, f):
        '''
        Frequnecy derivative of velocity as a function of frequency. 
        dvdf = dvdf(f)
        Units: [f] = Hz, [dvdf] = s
        '''        
        return self.mtotal*np.pi/(3*self.v(f)*self.v(f))

    def psi(self, freqs):
        '''
        Theoretical FD waveform phase as a function of frequency.
        Psi = Psi(f)
        Units: [f] = Hz, [Psi] = rad
        '''
        tmp = (3. / 128) * (np.pi * self.mchirp * freqs) ** (-5/3)
        return tmp - tmp[0] # set initial (at t0) phase to 0
    
    @property
    def merger_time(self):
        '''
        Theoretical Merger Time as a function of frequency.
        t = t(f)
        Units: [f] = Hz, [t] = s
        '''
        return lambda f: - (5./256) * self.mtotal / self.eta * self.v(f) ** (-8)

    @property
    def distance(self):
        '''
        Theoretical Distance as a function of frequency. 
        D = D(f)
        Units: [f] = Hz, [D] = km
        '''
        return lambda f: ((G*self.mtotal/msolsec*M_sun)* (np.pi * f) ** -2.)**(1/3) /1000 

    @property
    def energy(self):
        '''
        Orbital Energy as a function of frequency.
        E = E(f)
        Units: [f] = Hz, [E] = s
        '''
        return lambda f: -self.mtotal * self.eta * self.v(f) ** 2 / 2. #E_0(v) PN

    @property
    def dEdv(self):
        '''
        Velocity derivative of energy as a function of frequency. 
        dEdv = dEdv(f)
        Units: [f] = Hz, [dEdv] = s
        '''
        return lambda f: -self.mtotal * self.eta * self.v(f) #E'_0(v)  

    @property
    def flux(self):
        '''
        Energy Flux as a function of frequency.
        F = F(f)
        Units: [f] = Hz, [F] = -
        '''
        return lambda f: 32./5 * self.eta**2 * self.v(f)**10 #F_0(v) PN

    # variables related to resonance
    def dE(self, dt, fres):
        '''
        Energy transferred as a function of the time shift and the resonant frequency. Narrow resonance.
        Delta E = Delta E(dt,fres)
        Units: [dt] = s, [fres] = Hz, [Delta E] = s
        '''
        return -self.flux(fres)*dt # carefull with the - sign
        
    def dF(self, dt, fres):
        '''
        Energy-flux transferred as a function of the time shift and the resonant frequency. Narrow resonance and Delta F/F<<1.
        Delta F = Delta F(dt,fres)
        Units: [dt] = s, [fres] = Hz, [Delta F] = s**-1 (Delta F(f) = Delta F delta(f-fres) ; [delta(f)]=[f**-1]=s)
        '''
        return self.flux(fres)**2*dt/(self.dEdv(fres)*self.dvdf(fres))

    def flux_frac_sharp_fres(self, dt, fres):
        '''
        Flux fraction (evaluated at resonance) as a function of the time shift and the resonant frequency. Narrow resonance.
        Delta F(f=fres) / F = Delta F(f=fres) / F(dt,fres)
        Units: [dt] = s, [fres] = Hz, [Delta F (f=fres)/ F] = - 
        '''        
        return -1./(1.+self.dEdv(fres)*self.dvdf(fres)/(self.flux(fres)*dt))
    
    def deriv_PN(self, dEdf = lambda f: 0., extra_flux = lambda f: 0.):
        '''
        dt/df, dpsi/df evaluated at some state
        '''
        flux = lambda f: self.flux(f) 
        dEdv = lambda f: self.dEdv(f)
        def deriv_func(state, f):
            t, psi = state
            dt = -(self.dvdf(f) * dEdv(f) + dEdf(f)) / (flux(f) + extra_flux(f))
            dpsi = 2*np.pi*t
            return dt, dpsi
        return deriv_func


################################################################################
############################ Energy transfer ###################################
################################################################################


# Energy
def dEdf_box(fres, width, dE):
    '''
    Top hat function
    '''
    return lambda f: dE / width if np.abs(f-fres) < width / 2 else 0.

# Energy-flux
def extra_flux_Delta(fres, width, dF):
    '''
    Delta function centered at the resonant frequency
    '''
    return [self.flux(fres)*dt/(self.dEdv(fres)*self.dvdf(fres)) if f==f_res else lambda f: 0.] # it is not possible to build a delta function

def extra_flux_box0(fres, width, dF):
    '''
    Top hat function. (if,else)
    '''
    return lambda f: dF / width if np.abs(f-fres) < width / 2 else 0.

def extra_flux_box(fres, width, dF):
    '''
    Top hat function. (np.where)
    '''
    return lambda f: (dF/width) * (np.where(f < (fres - width/2), 0., 1)-np.where(f < (fres + width/2), 0., 1))

def extra_flux_DHO(fres, width, dF):
    '''
    Driven Damped Harmonic Oscillator equation solution
    '''
    return lambda f: dF * width * 2 / np.pi * f ** 2 / ((width * f) ** 2 + (fres ** 2 - f ** 2) ** 2)


################################################################################
#################### PHASE DIFFERENCES. dPsi ###################################
################################################################################

# Time difference
def t_shift(dt, fres, freqs):
    '''
    Step function
    '''
    return np.where(freqs < fres, 0., dt)

def t_shift_tanh(dt, fres, freqs, sharpness=0.00001):
    '''
    Hyperbolic tangent
    '''
    return dt*(np.tanh((freqs-fres)/sharpness)+1)/2


# Phase difference
def phase_diff_t_shift(dt, fres, freqs):
    '''
    Ramp function
    '''
    return np.where(freqs < fres, 0., 2.*np.pi*dt*(freqs-fres))

def phase_shift(dphi, fres, freqs):
    '''
    Step function
    '''
    return np.where(freqs < fres, 0., dphi)

def solve_ode(dEdf = lambda f: 0., extra_flux = lambda f: 0., m1=1.4, m2=1.4, f_low=15, tlen=16, srate=16384):
    '''
    Integrate dt/df and dpsi/df. 
    frequencies=(f_low, srate/2)
    '''
    
    def add_zeros(x): # Add zeros in 0<f<f_low
        """Add zeros to a vector in 0<f<f_low"""
        x_aux = np.zeros(1+tlen*srate//2)
        x_aux[int(f_low*tlen):] = x
        return x_aux
        
    binary = System(m1, m2)

    freqs = np.arange(f_low*tlen, tlen*srate//2+1)/tlen
    
    psi = binary.psi(freqs)

    IC = np.array([binary.merger_time(f_low), 0.]) # Initial Conditions
    
    deriv0 = binary.deriv_PN(dEdf = lambda f: 0., extra_flux = lambda f: 0.)
    soln = odeint(deriv0, IC, freqs, rtol=1e-13, atol=1e-13) 

    deriv_R =binary.deriv_PN(dEdf = dEdf, extra_flux = extra_flux)
    soln_R = odeint(deriv_R, IC, freqs, rtol=1e-13, atol=1e-13) #[t_R,phase_R]

    dt = add_zeros(soln_R[:,0] - soln[:,0])
    dPsi = add_zeros(soln_R[:,1] - soln[:,1])
    #Check_1, Check_2 = np.polyfit(2*np.pi*freqs, soln[:,1]-psi,1) # our solution-theoretical phase 

    t = add_zeros(soln[:,0])
    Psi = add_zeros(soln[:,1])

    t_R = add_zeros(soln_R[:,0])
    Psi_R = add_zeros(soln_R[:,1])
    return dPsi, dt, t,Psi,t_R,Psi_R

def dt_dpsi(dPsi, freqs, fres, f_low=15, tlen=16, srate=16384):
    '''
    Linear fit over the phase difference. Extract the time and phase difference
    '''
    idx = int(fres * tlen) # index at resonance
    assert np.abs(freqs[idx] - fres) < 0.01
    
    idx2 = idx + 10 * tlen # index a bit after the resonance (at f=fres+30Hz)      
    return np.polyfit(2*np.pi*freqs[idx2:], dPsi[idx2:], 1) #Delta_t, Delta_phi



################################################################################
#################### MATCH-FILTER ###################################
################################################################################


class Filter():
    '''
    Match Filtering
    '''
    def __init__(self, f_low, f_high, tlen, srate, snr, approximant, detector, m1=1.4, m2=1.4, dL=100):

        self.f_low = f_low
        self.f_high = f_high
        self.tlen = tlen
        self.srate = srate

        self.snr_var = snr
        self.dL = dL
        self.approximant = approximant

        self.m1 = m1
        self.m2 = m2

        # Initialize with the detector
        self._detector = None  # Internal variable to store the detector
        self.detector = detector  # This triggers the property setter and updates PSD

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value
        self.update_psd()  # Automatically update PSD when detector changes

    def update_psd(self):
        """Updates PSD and frequencies based on the selected detector."""
        nyquist_freq = self.srate // 2  # Nyquist frequency based on sample rate
        f_len = int(1+self.tlen*self.f_high)
        # Load PSD
        if isinstance(self.detector, str):
            script_directory = os.path.dirname(os.path.abspath(__file__))
            psd = from_txt(os.path.join(script_directory,f'noise_curves/{self.detector}.txt'), int(1 + self.tlen * self.srate // 2), 1. / self.tlen, low_freq_cutoff=self.f_low, is_asd_file=True)
            freqs = psd.sample_frequencies.numpy()
            if freqs[-1] < nyquist_freq:
                # Pad after the Nyquist frequency
                self.freqs = np.arange(0, self.tlen*self.srate//2+1)/self.tlen
                psd0 = np.zeros_like(self.freqs)
                psd0[:len(psd)] = psd.data
                self.psd=FrequencySeries(psd0, delta_f=1/self.tlen)
            else:
                # Use original PSD if it covers up to Nyquist frequency
                self.psd = psd
                self.freqs = freqs

                
        else:
            tmp_psd = np.zeros(int(1 + self.tlen * self.srate // 2))
            tmp_psd[:f_len] = self.detector(f_len, 1./self.tlen, low_freq_cutoff = self.f_low).data[:]
            self.psd = FrequencySeries(tmp_psd, delta_f = 1/self.tlen)
            #self.psd = self.detector(int(1 + self.tlen * self.srate // 2), 1. / self.tlen, low_freq_cutoff=self.f_low).data[:]
            self.freqs = self.psd.sample_frequencies.numpy()  

        self.inv_psd = np.where(self.psd > 0, 1.0 / self.psd, 0.0)

        # Set up kwargs for template matching
        self.kwargs = {
            'psd': self.psd,
            'low_frequency_cutoff': self.f_low,
            'high_frequency_cutoff': self.f_high  # Set to Nyquist for sample rate
        }
        
        # Normalize template
        self.tmplt = self.template()
        self.tmplt_snr = self.template_snr()

        #tmplt_norm = tmplt / sigma(tmplt, **self.kwargs)
        #self.tmplt = tmplt_norm
    
    def template(self, m1=1.4, m2=1.4, distance=100):
        '''
        Return a frequency-domain waveform with the given masses.
        All waveforms are truncated at f_high.
        This could be replaced with a lowpass filter to simulate the lack of information about the merger.
        The luminosity distance is fixed
        '''
        self.dL = distance
        hp, _ = get_fd_waveform(approximant=self.approximant,
                                    mass1=self.m1, mass2=self.m2,
                                    delta_f = 1/self.tlen,
                                    distance = self.dL,
                                    f_lower = self.f_low, f_upper=self.f_high)
        hp.resize(int(1+self.tlen*self.srate//2))
        hp[int(self.tlen*self.f_high):] = 0.

        self.snr_var = sigma(hp, **self.kwargs) 
        return hp

    def template_snr(self, m1=1.4, m2=1.4, snr=100):
        '''
        Return a frequency-domain waveform with the given masses.
        All waveforms are truncated at f_high.
        This could be replaced with a lowpass filter to simulate the lack of information about the merger
        The SNR is fixed
        '''
        snr_guess = 1
        distance_guess = snr_guess * self.dL / snr
        
        hp = self.template(m1,m2,distance_guess)

        tmplt_norm = hp / sigma(hp, **self.kwargs) 
        
        self.tmplt_snr = tmplt_norm * snr

        self.snr_var = sigma(hp, **self.kwargs) 
        return self.tmplt_snr

        
    def match_PyCBC(self, dPsi):
        '''
        Compute the match among two waveforms differing by a phase factor
        '''
        h = self.tmplt
        s = h*np.exp(-1.j*dPsi)
        return match(h, s, **self.kwargs)[0]

    def J(self):
        '''
        Noise momentum integrals
        Independent of the phase difference
        in = sigmasq((2*np.pi*self.freqs)**((7-n)/6)*tmplt, **self.kwargs)
        '''
        h_tmplt = self.tmplt / sigma(self.tmplt, **self.kwargs)
        i7 = sigmasq(h_tmplt, **self.kwargs)
        i4 = sigmasq(np.sqrt(2*np.pi*self.freqs)*h_tmplt, **self.kwargs)
        i1 = sigmasq(2*np.pi*self.freqs*h_tmplt, **self.kwargs)

        return i7, i4, i1

    def J_dPsi(self, dPsi):
        '''
        Noise momentum integrals
        Components dependent on the phase difference
        idYn = sigmasq((2*np.pi*self.freqs)**((7-n)/6)*np.sqrt(dPsi)*tmplt, **self.kwargs)
        dY=np.sqrt(dPsi)
        idYsqn = sigmasq((2*np.pi*self.freqs)**((7-n)/6)*dPsi*tmplt, **self.kwargs)
        '''
        h_tmplt = self.tmplt / sigma(self.tmplt, **self.kwargs)
        idY7 = np.sign(dPsi)[-1]*sigmasq(h_tmplt*np.sqrt(np.abs(dPsi)), **self.kwargs)
        idY4 = np.sign(dPsi)[-1]*sigmasq(np.sqrt(2*np.pi*self.freqs)*np.sqrt(np.abs(dPsi))*h_tmplt, **self.kwargs)
        idYsq7 = sigmasq(dPsi*h_tmplt, **self.kwargs)

        return idY7, idY4, idYsq7

    def match_approx(self, dPsi): #quad
        '''
        Compute the quadratic approximation of the match for a given phase difference
        '''
        i7, i4, i1 = self.J()
        idY7, idY4, idYsq7 = self.J_dPsi(dPsi)
        a, b = i1-i4*i4, idY4-i4*idY7
        return 1-(idYsq7-idY7*idY7 - b*b/a)/2

    def match_dx(self, fres, dPsi, dx_arr):
        '''
        Match PyCBC and approx for some phase difference function dPsi(dx, fres, bns.freqs) paramterized by dx parameter
        Typically, the phase difference can be either a step or a ramp function centered at f=fres. 
        match_vec[0]= match PyCBC, match_vec[1]= match approx for an array of dx.
        dx parametrices the family of dPsi. It can indicate for example how strong is a time or phase shift
        '''        
        # Vectorized match calculations
        match_PyCBC_arr = np.vectorize(lambda dx: self.match_PyCBC(dPsi(dx, fres, self.freqs)))(dx_arr)
        match_approx_arr = np.vectorize(lambda dx: self.match_approx(dPsi(dx, fres, self.freqs)))(dx_arr)
        match_finer_arr = np.vectorize(lambda dx: self.match_finer(dPsi(dx, fres, self.freqs)))(dx_arr)

        return match_PyCBC_arr, match_approx_arr, match_finer_arr

    def match_approx_error(self, dPsi):
        '''
        Compute relative error among the PyCBC match function and the approximated match
        '''
        match_PyCBC = self.match_PyCBC(dPsi)
        match_approx = self.match_approx(dPsi)
        return np.abs(match_PyCBC-match_approx)/match_PyCBC
    
    def match_approx_error_finer(self, dPsi):
        '''
        Compute relative error among the PyCBC match function and the approximated match
        '''
        match_finer = self.match_finer(dPsi)
        match_approx = self.match_approx(dPsi)
        return np.abs(match_finer-match_approx)/match_finer

    def dx_fres_fix_error(self, error, fres_arr, dPsi):
        '''
        Time shift as a function of resonant frequency for different match approximation errors
        '''
        if dPsi==phase_shift:
            a1=1e-4
            a2=10
        else:
            a1=1e-6
            a2=1e-1  
            
        dx_list = []
        for fres in fres_arr:
            try:
                dx_list.append(bisect(lambda dx: self.match_approx_error_finer(dPsi(dx, fres, self.freqs)) - error, -a2, -a1))
            except:
                print(fres)
                dx_list.append(bisect(lambda dx: self.match_approx_error_finer(dPsi(dx, fres, self.freqs)) - error, -2**np.pi, -1e-8))
        dx_arr = np.array(dx_list)
        return dx_arr

    def dx_quad_func(self, MM_detect, dPsi, dx_seed):
        '''
        MM_detect : Mismatch threshold for detectability
        dPsi : phase difference function : dPsi(dx, fres, freqs)
        dx_seed : guess for the perturbation (should be small to gurantee that we keep on the quadratic regime)
        '''
        return lambda fres: dx_seed*np.sqrt(MM_detect/(1-self.match_approx(dPsi(dx_seed, fres, self.freqs))))

        
    def detectable_dx(self, MM_detect, fres_arr, dPsi, dx_seed, dx_cut_arr=None): 
        '''
        For a given resonant frequency, this funtion allows to find the amplitude of the resonant corresponding to the detectability threshold
        '''
        dx_guess_func = self.dx_quad_func(MM_detect, dPsi, dx_seed)
        dx_guess_arr = np.array([dx_guess_func(fres) for fres in fres_arr])    
        # Find phi such that MM = MM_detect
        def MM_root(dx, fres):
            if dx_cut_arr is None:
                return 1 - self.match_finer(dPsi(dx, fres, self.freqs)) - MM_detect
            else:
                dx_lim = abs(dx_cut_arr[int(fres//fres_arr[-1]*len(fres_arr)-1)])
                if abs(dx)>dx_lim: 
                    return 1 - self.match_finer(dPsi(dx, fres, self.freqs)) - MM_detect
                else:
                    return 1 - self.match_approx(dPsi(dx, fres, self.freqs)) - MM_detect

        if dPsi==phase_shift:
            a1=1e-4
            a2=10
        else:
            a1=1e-6
            a2=5e-1        
        # Use the approximated prediction as a guess to bracket the exact value
        dx_list = []
        for fres in fres_arr:
            a=dx_guess_func(fres)
            if a>0:
                try:
                    dx_list.append(bisect(lambda dx: MM_root(dx,fres),0.1*a, 2*a))    
                except:
                    print(fres)
                    dx_list.append(bisect(lambda dx: MM_root(dx,fres), a1, a2))
            else:
                try:
                    dx_list.append(bisect(lambda dx: MM_root(dx,fres),2*a, 0.1*a)) 
                except:
                    print(fres)
                    dx_list.append(bisect(lambda dx: MM_root(dx,fres), -a2, -a1))
        dx_arr = np.array(dx_list)
        
        return dx_arr, dx_guess_arr

    def detectable_dx_quad_sigma(self, sigma, fres_arr, dPsi, dx_seed): 
        '''
        
        '''
        rho_sq = sigmasq(self.tmplt, **self.kwargs)
        MM_detect = 1/(2*rho_sq*sigma**2)
        dx_quad_func = self.dx_quad_func(MM_detect, dPsi, dx_seed)
        return np.array([dx_quad_func(fres) for fres in fres_arr])

    def read2023_fixed_dL(self,f): 
        '''
        Read 2023 approximation. 
        Phase shift detectability threshold fixing the luminosity distance
        '''
        index_arr = f * self.tlen
        return -np.array([(np.sqrt(self.psd/self.freqs)/(2*np.abs(self.tmplt)))[int(i)] for i in index_arr]) #[:f_high:idx_spacing=t_len]

    def read2023_fixed_snr(self, f,snr=100.): 
        '''
        Read 2023 approximation.
        Phase shift detectability threshold fixing the SNR
        '''
        index_arr = f * self.tlen
        return -np.array([(np.sqrt(self.psd/f)/(2*snr*np.abs(self.tmplt)/ sigma(self.tmplt, **self.kwargs)))[int(i)] for i in index_arr]) #[:f_high:idx_spacing=t_len]

    def snr(self, tmplt):
        '''
        SNR: inner product of the template with itself weighted by the noise
        '''
        return sigma(tmplt, **self.kwargs)
    
    def matched_filter(self, tmplt1, tmplt2):
        '''
        Overlap normalized from 0 to 1
        '''
        return matched_filter(tmplt1, tmplt2,**self.kwargs)/np.sqrt(self.snr(tmplt1)*self.snr(tmplt2))
    
    def rho(self, tmplt1, tmplt2):
        '''
        Weighted SNR 
        <h,d>/sqrt<h,h>
        '''
        return matched_filter(tmplt1, tmplt2,**self.kwargs)/self.snr(tmplt1)
    
    def rho_abs_roll_split(self, dt, fres, shift):
        '''
        Absolute value of the overlap.
        Time roll
        '''
        h = self.tmplt

        dPsi = phase_diff_t_shift(dt, fres, self.freqs)
        s = h*np.exp(1.j*dPsi)

        sa = s * (1. - np.heaviside(self.freqs - fres, 0.5))
        sb = s * np.heaviside(self.freqs - fres, 0.5)

        rho_abs = np.abs(self.roll_ts(self.rho(h, s), shift)[:2*shift])

        rhoa_abs = np.abs(self.roll_ts(self.rho(h, sa), shift)[:2*shift])

        rhob_abs = np.abs(self.roll_ts(self.rho(h, sb), shift)[:2*shift])

        tx = rho_abs.sample_times
        return tx, rho_abs, rhoa_abs, rhob_abs

    def overlap_func(self, tmplt1, tmplt2):
        '''
        Absolute value of the overlap.
        '''
        return lambda dt: np.abs(overlap_cplx(tmplt1.cyclic_time_shift(dt),tmplt2,**self.kwargs))
    
    def match_finer(self, dPsi):
        '''
        Optimized match computation.
        max_dt(abs(overlap(dt)))
        '''
        h = self.tmplt
        s = h*np.exp(-1.j*dPsi)

        dt_fit, dpsi_fit=np.polyfit(2*np.pi*self.freqs, dPsi, 1) #fit to line to fix bounds (assumin dpsi is small)
        
        t_bound = 10*np.abs(dt_fit)

        return -minimize_scalar(lambda dt: -self.overlap_func(h, s)(dt),bounds=(-t_bound, t_bound)).fun 
    
    def overlap_roll(self, tmplt1, tmplt2, shift, srate_finer):
        '''
        Overlap normalized
        Time roll
        shift = number of samples of the time vector
        srate_finer = sampling rate applied for the optimized match computation
        '''
        t_roll = np.arange(-shift+1, shift) / srate_finer

        overlap_vec = np.vectorize(lambda dt:self.overlap_func(tmplt1, tmplt2)(dt))
        return t_roll, overlap_vec(t_roll)

    def roll_ts(self, ts, shift): 
        '''
        Roll time series
        shift = number of samples to roll
        shift = int(t_shift * srate)
        '''
        rolled_data = np.roll(ts.data, shift)[:2*shift]
        return TimeSeries(rolled_data, delta_t=ts.delta_t, epoch=ts.start_time+self.tlen-shift/self.srate)
