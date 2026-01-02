from dataclasses import dataclass, field
from typing import Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import jax
from pulsedjax.utilities import get_sk_rn, do_fft, do_ifft, MyNamespace, do_interpolation_1d, generate_random_continuous_function, get_com



    

class SpectralAmplitude:
    """ A base for all spectral amplitudes. """
        

class SpectralPhase:
    """ A base for all spectral phases"""


@dataclass
class GaussianAmplitude(SpectralAmplitude):
    amplitude: Union[np.ndarray,tuple,float,int]
    """ scales the gaussian by a factor """

    central_frequency: Union[np.ndarray,tuple,float,int]
    """ the position of the maximum on the frequency axis """

    fwhm: Union[np.ndarray,tuple,float,int]
    """ the Full-Width-Half-Maximum of the gaussian """

    p: Union[np.ndarray,tuple,float,int,None] = None
    """ defines a super-gaussian (needs to be bigger or equal to one), if None a normal gaussian is assumed. """


@dataclass
class LorentzianAmplitude(SpectralAmplitude):
    amplitude: Union[np.ndarray,tuple,float,int]
    """ scales the lorentzian by a factor """

    central_frequency: Union[np.ndarray,tuple,float,int]
    """ the position of the maximum on the frequency axis """

    fwhm: Union[np.ndarray,tuple,float,int]
    """ the Full-Width-Half-Maximum of the lorentzian """

    p: Union[np.ndarray,tuple,float,int,None] = None
    """ defines a super-lorentzian (needs to be bigger or equal to one), if None a normal lorentzian is assumed. """


@dataclass
class PolynomialPhase(SpectralPhase):
    central_frequency: Union[float,int,None]
    """ the central frequency, if None it will be calculated from the spectral amplitude. """

    coefficients: Union[np.ndarray,tuple,float,int]
    """ the taylor coefficients in fs, need to include all orders. (e.g. for a TOD pulse the input needs to be (0,0,0,TOD)) """


@dataclass
class SinusoidalPhase(SpectralPhase):
    amplitude: Union[np.ndarray,tuple,float,int]
    """ the amplitude of the sinusoid (in units of 2*pi) """

    central_frequency: Union[np.ndarray,tuple,float,int]
    """ the base-shift of the sinusoid (in PHz) """

    periodicity: Union[np.ndarray,tuple,float,int]
    """ the periodicity of the sinusoid (in fs) """

    phase_shift: Union[np.ndarray,tuple,float,int]
    """ the phase of the sinusoid """


@dataclass
class RandomPhase(SpectralPhase):
    """
    Generates a random continuous phase in the frequency domain. 

    """
    number_of_points: int = 10
    minval_phase: Union[float,int] = -0.5*np.pi
    maxval_phase: Union[float,int] = 0.5*np.pi
    sigma: Union[float,int] = 10


@dataclass
class CustomPulse:
    frequency: np.ndarray
    """ the frequency axis """

    amplitude: Union[np.ndarray,SpectralAmplitude]
    """ the spectral amplitude """

    phase: Union[np.ndarray,SpectralPhase]
    """ the spectral phase """



@dataclass
class TemporalAmplitude:
    gaussian_or_lorentzian: str
    """ defines the envelope shape, needs to be gaussian or lorentzian. """

    amplitude: Union[float, int]
    """ the amplitude  """

    duration: Union[float, int]
    """ the duration """

    central_frequency: Union[float, int]
    """ the central_frequency """

    p: Union[float, int, None] = None
    """ the super-gaussian/lorentzian exponent """



@dataclass
class MultiPulse:
    gaussian_or_lorentzian: Union[list,tuple,np.ndarray,str]
    """ only "G" for gaussian or "L" for lorentzian are accepted """

    amplitude: Union[list,tuple,np.ndarray,float,int]
    """ the relative scale of the envelopes """

    delay: Union[list,np.ndarray,tuple,float,int]
    """ the relative delay (not position) of each pulse, has to be one less than the total number of pulses """

    duration: Union[list,tuple,np.ndarray,float, int]
    """ the FWHM of each envelope in the time domain """

    central_frequency: Union[list,tuple,np.ndarray,float,int]
    """ the central frequency of each pulse """

    p: Union[list,tuple,np.ndarray,float,int,None]
    """ the super-gaussian/lorentzian of each pulse in the time domain """

    phase: Union[list[SpectralPhase],tuple[SpectralPhase], SpectralPhase]
    """ the spectral phase of each pulse """

    envelope: Union[list[TemporalAmplitude], tuple[TemporalAmplitude], TemporalAmplitude] = field(init=False)
    """ is constructed from the given attributes """
    
    def __post_init__(self):
        g_or_l = np.squeeze(np.asarray([self.gaussian_or_lorentzian]))
        amp = self.amplitude
        dt = self.duration
        cf = self.central_frequency
        p = self.p

        if p==None:
            p=np.broadcast_to(1,np.shape(np.asarray([amp])))
        check_and_correct_if_scalar((amp,dt,cf,p))
        assert np.size(amp)==np.size(dt)==np.size(cf)==np.size(p)==np.size(g_or_l)

        env_list = []
        n = len(amp)
        for i in range(n):
            g_or_l_val = g_or_l[i]
            if g_or_l_val=="G":
                g_or_l_val = "gaussian"
            elif g_or_l_val=="L":
                g_or_l_val="lorentzian"
            else:
                raise ValueError
            env_list.append(TemporalAmplitude(g_or_l_val,amp[i],dt[i],cf[i],p[i]))

        self.envelope = env_list






def check_and_correct_if_scalar(parameters):
    if all([type(j)!=i for i in (int,float) for j in parameters]):
        temp = parameters
    else:
        temp = []
        for param in parameters:
            temp.append([np.squeeze(np.asarray([param]))])
    return temp


class MakePulse:
    """ 
    Is supposed to generate pulses based on input parameters from GaussianAmplitude, PolynomialPhase, SinusoidalPhase, RandomPhase, CustomPulse or MultiPulse.

    Attributes:
        N (int): the number of points on the time/frequency-grid
        f_max (float): the frequency range which is used. Extends from -f_max to +f_max (so it should actually be named f_max?)

    """
    def __init__(self, N=256, f_max=None):
        self.N = N
        self.f_max = f_max


    def gaussian(self, x, amp, fwhm, shift, p):
        return amp*np.exp(-np.log(2)*(4*(x-shift)**2/fwhm**2)**p)
    
    def lorentzian(self, x, amp, fwhm, shift, p):
        return amp/(1+np.abs(2*(x-shift)/fwhm)**(2*p))

    
    def generate_gaussian_lorentzian_amplitude(self, x, parameters, amp_func):
        amp, fwhm, shift, p = parameters.amplitude, parameters.fwhm, parameters.central_frequency, parameters.p

        if p==None:
           p = np.broadcast_to(1, np.shape(amp))

        amp, fwhm, shift, p = check_and_correct_if_scalar((amp, fwhm, shift, p))
        assert len(amp)==len(fwhm)==len(shift)==len(p)

        a = 0
        for i in range(len(amp)):
            a = a + amp_func(x, amp[i], fwhm[i], shift[i], p[i])
        return a
    

    def generate_polynomial_phase(self, frequency, parameters, amp_f):
        central_frequency, coefficients = parameters.central_frequency, np.squeeze(np.asarray([parameters.coefficients]))
        if central_frequency==None:
            idx_f0 = get_com(amp_f, np.arange(len(self.frequency))).astype(int)
            central_frequency = self.frequency[idx_f0]
        
        n = len(coefficients)
        phase = 0
        for i in range(n):
            phase = phase + coefficients[i]*(2*np.pi*(frequency-central_frequency))**i
        return phase
    
    
    def generate_sinusoidal_phase(self, frequency, phase_parameters):
        a, f0, b,c = phase_parameters.amplitude, phase_parameters.central_frequency, phase_parameters.periodicity, phase_parameters.phase_shift
        a, f0, b,c = check_and_correct_if_scalar((a,f0,b,c))
        assert len(a)==len(f0)==len(b)==len(c)

        phase = 0
        for i in range(len(a)):
            phase = phase + a[i]*np.sin(2*np.pi*b[i]*(frequency-f0[i])-c[i])
        return 2*np.pi*phase
    

    def get_spectral_phase(self, phase_parameters, amp_f):
        if isinstance(phase_parameters, PolynomialPhase):
            phase = self.generate_polynomial_phase(self.frequency, phase_parameters, amp_f)

        elif isinstance(phase_parameters, SinusoidalPhase):
            phase = self.generate_sinusoidal_phase(self.frequency, phase_parameters)

        elif isinstance(phase_parameters, RandomPhase):
            N_points, sigma = phase_parameters.number_of_points, phase_parameters.sigma
            min_val, max_val = phase_parameters.minval_phase, phase_parameters.maxval_phase

            idx = np.where(amp_f>0.01)
            xf = self.frequency[idx[0][0]], self.frequency[idx[0][-1]]
            phase = generate_random_continuous_function(jax.random.PRNGKey(np.random.randint(0,1e9)), N_points, self.frequency, 
                                                        min_val, max_val, gaussian_filter1d(amp_f, sigma=sigma), forced_vals=xf, extrap=True)
            
        elif isinstance(phase_parameters, CustomPulse):
            if isinstance(phase_parameters.phase, SpectralPhase):
                phase = self.get_spectral_phase(phase_parameters.phase, amp_f)
            else:
                phase = do_interpolation_1d(self.frequency, phase_parameters.frequency, phase_parameters.phase)

        else:
            raise NotImplementedError(f"Phase type {phase_parameters} is not implemented.")

        return phase
    


    def get_spectral_amplitude(self, amp_parameters):
        if isinstance(amp_parameters, GaussianAmplitude):
            amp_f = self.generate_gaussian_lorentzian_amplitude(self.frequency, amp_parameters, self.gaussian)
        
        elif isinstance(amp_parameters, LorentzianAmplitude):
            amp_f = self.generate_gaussian_lorentzian_amplitude(self.frequency, amp_parameters, self.lorentzian)
        
        elif isinstance(amp_parameters, MultiPulse):
            raise ValueError("This logic path shouldnt be reachable.")

        elif isinstance(amp_parameters, CustomPulse):
            if isinstance(amp_parameters.amplitude, SpectralAmplitude):
                amp_f = self.get_spectral_amplitude(amp_parameters.amplitude)
            else:
                amp_f = do_interpolation_1d(self.frequency, amp_parameters.frequency, amp_parameters.amplitude)
        else:
            raise NotImplementedError(f"Amplitude type {amp_parameters} is not implemented.")

        return amp_f
    

    def get_multi_pulse(self, parameters, shift):
        pulse_t_total = 0

        n = len(shift)
        for i in range(n):
            env_t = parameters.envelope[i]
            phase_f = parameters.phase[i]
            deltat = shift[i]

            assert env_t.gaussian_or_lorentzian=="gaussian" or env_t.gaussian_or_lorentzian=="lorentzian"
            amp_t = getattr(self, env_t.gaussian_or_lorentzian)(self.time, env_t.amplitude, env_t.duration, deltat, env_t.p)
            pulse_t = amp_t*np.exp(1j*2*np.pi*env_t.central_frequency*self.time)
            pulse_f = do_fft(pulse_t, self.sk, self.rn)

            phase_f = self.get_spectral_phase(phase_f, np.abs(pulse_f))
            pulse_f = pulse_f*np.exp(1j*phase_f)
            pulse_t = do_ifft(pulse_f, self.sk, self.rn)
            pulse_t_total = pulse_t_total + pulse_t
        return pulse_t_total
    
        
        
    def init_generation(self, parameters):
        if isinstance(parameters, tuple)==True:
            amplitude, phase = parameters
            assert isinstance(amplitude, SpectralAmplitude) and isinstance(phase, SpectralPhase)

            if self.f_max == None:
                f = amplitude.fwhm + amplitude.central_frequency
                fmax = np.maximum(np.abs(np.min(f)), np.abs(np.max(f)))
                fmin, fmax = -4*fmax, 4*fmax
            else:
                fmin, fmax = -1*self.f_max, self.f_max

        elif isinstance(parameters, CustomPulse):
            assert self.f_max != None, "Please predefine a suitable f_max for CustomPulse."
            fmin, fmax = -1*self.f_max, self.f_max
            parameters = parameters, parameters

        elif isinstance(parameters, MultiPulse):
            if self.f_max == None:
                Delta_t = 2*(np.max(parameters.delay) - np.min(parameters.delay))
                df = 1/Delta_t
                self.f_max = self.N*df
            else:
                pass
            
            fmin, fmax = -1*self.f_max, self.f_max

        else:
            raise TypeError(f"The input {parameters} has to be a tuple[SpectralAmplitude, SpectralPhase], CustomPulse or MultiPulse. Not {type(parameters)}.")

        assert type(fmax)==int or type(fmax)==float, "There is a type issue with f_max."
        self.frequency = np.linspace(fmin, fmax, self.N)
        self.df = np.mean(np.diff(self.frequency))
        self.time = np.fft.fftshift(np.fft.fftfreq(self.N, self.df))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        return parameters



    def generate_pulse(self, parameters):
        """
        Returns a pulse in time and frequency domain, on a discretized grid,

        Args:
            parameters (tuple[SpectralAmplitude, SpectralPhase], CustomPulse, MultiPulse): the input parameters

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, jnp.array], the pulse and correspnding axes (time, pulse_t, frequency, pulse_f)
        """
        
        parameters = self.init_generation(parameters)

        if isinstance(parameters, MultiPulse): 
            shift = np.concatenate((np.zeros(1), parameters.delay))
            shift = shift - np.mean(shift)
            assert len(parameters.envelope)==len(parameters.phase)==len(shift)
            pulse_t = self.get_multi_pulse(parameters, shift)
            pulse_f = do_fft(pulse_t, self.sk, self.rn)
            
        else:
            amplitude, phase = parameters
            amp_f = self.get_spectral_amplitude(amplitude)
            phase = self.get_spectral_phase(phase, amp_f)
            pulse_f = amp_f*np.exp(1j*phase)
            pulse_t = do_ifft(pulse_f, self.sk, self.rn)

        self.pulses = MyNamespace(time=self.time, frequency=self.frequency, pulse_t=pulse_t, pulse_f=pulse_f)
        return self.time, pulse_t, self.frequency, pulse_f
    




    def plot_envelopes(self):
        fig = plt.figure(figsize=(9,4))

        plt.subplot(1,2,1)
        plt.plot(self.pulses.time, np.abs(self.pulses.pulse_t))
        plt.xlabel("Time [fs]")
        plt.ylabel("Amplitude [arb. u.]")

        plt.subplot(1,2,2)
        plt.plot(self.pulses.frequency, np.abs(self.pulses.pulse_f))
        plt.xlabel("Frequency [pHz]")
        plt.ylabel("Amplitude [arb. u.]")

        plt.show()
