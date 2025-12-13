from dataclasses import dataclass, field
from typing import Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import jax
from src.utilities import get_sk_rn, do_fft, do_ifft, MyNamespace, do_interpolation_1d, generate_random_continuous_function, get_com


def check_dataclass_types(self):
    """ Makes sure the correct types are used in the dataclasses. """
    for (name, field) in self.__dataclass_fields__.items():
        field_type = field.type
        if not isinstance(self.__dict__[name], field_type):
            current_type = type(self.__dict__[name])
            raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")

class SpectralAmplitude:
    """ A base for all spectral amplitudes. """
    def __post_init__(self):
        check_dataclass_types(self)
        

class SpectralPhase:
    """ A base for all spectral phases"""
    def __post_init__(self):
        check_dataclass_types(self)


@dataclass
class GaussianAmplitude(SpectralAmplitude): # can be used for single or multiple gaussians
    """ 
    Defines a Gaussian envelope. Multiple parameters can be provided per attribute. Final gaussians will be added up.

    Attributes:
        amplitude: scales the gaussian by a factor
        central_frequency: the position of the maximum on the frequency axis
        fwhm: the Full-Width-Half-Maximum of the gaussian
        p: defines a super-gaussian (needs to be bigger or equal to one), if None a normal gaussian is assumed.
    """
    amplitude: Union[np.ndarray,tuple,float,int]
    central_frequency: Union[np.ndarray,tuple,float,int]
    fwhm: Union[np.ndarray,tuple,float,int]
    p: Union[np.ndarray,tuple,float,int,None] = None


@dataclass
class LorentzianAmplitude(SpectralAmplitude): # can be used for single or multiple lorentzians
    """ 
    Defines a Lorentzian envelope. Multiple parameters can be provided per attribute. Final lorentzians will be added up.

    Attributes:
        amplitude: scales the lorentzian by a factor
        central_frequency: the position of the maximum on the frequency axis
        fwhm: the Full-Width-Half-Maximum of the lorentzian
        p: defines a super-lorentzian (needs to be bigger or equal to one), if None a normal lorentzian is assumed.
    """
    amplitude: Union[np.ndarray,tuple,float,int]
    central_frequency: Union[np.ndarray,tuple,float,int]
    fwhm: Union[np.ndarray,tuple,float,int]
    p: Union[np.ndarray,tuple,float,int,None] = None



@dataclass
class PolynomialPhase(SpectralPhase):
    """ 
    Defines a polynomial phase in the frequency domain.

    Attributes:
        central_frequency: the central frequency, if None it will be calculated from the spectral amplitude.
        coefficients: the taylor coefficients in fs, need to include all orders. (e.g. for a TOD pulse the input needs to be (0,0,0,TOD))
    
    """
    central_frequency: Union[float,int,None]
    coefficients: Union[np.ndarray,tuple,float,int]


@dataclass
class SinusoidalPhase(SpectralPhase):
    """
    Defines a sinusoidal phase in the frequency domain. Multiple parameters can be provided per attribute. Will cause addition of sinusoids.

    Attributes:
        amplitude: the amplitude of the sinusoid (in units of 2*pi)
        central_frequency: the base-shift of the sinusoid (in PHz)
        peridoicity: the periodicity of the sinusoid (in fs)
        phase_shift: the phase of the sinusoid
    
    """
    amplitude: Union[np.ndarray,tuple,float,int]
    central_frequency: Union[np.ndarray,tuple,float,int]
    periodicity: Union[np.ndarray,tuple,float,int]
    phase_shift: Union[np.ndarray,tuple,float,int]


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
    """
    Defines an arbitrary pulse via arrays. Or a mixture of a discretized and parametrized amplitude or phase.

    Attributes:
        frequency: the frequency axis
        amplitude: the spectral amplitude, can be an array or a SpectralAmplitude
        phase: the spectral phase, can be an array or a SpectralPhase
    """
    frequency: np.ndarray
    amplitude: Union[np.ndarray,SpectralAmplitude]
    phase: Union[np.ndarray,SpectralPhase]

    def __post_init__(self):
        check_dataclass_types(self)




class TemporalAmplitude:
    """ 
    Defines a pulse envelope in the time domain. Is only used/accepted by MultiPulse.

    Attributes:
        gaussian_or_lorentzian: defines the envelope shape, needs to be gaussian or lorentzian.
        ampitude: the amplitude 
        duration: the duration
        central_frequency: the central_frequency
        p: the super-gaussian/lorentzian exponent
    
    """
    gaussian_or_lorentzian: str
    amplitude: Union[float, int]
    duration: Union[float, int]
    central_frequency: Union[float, int]
    p: Union[float, int, None] = None

    def __post_init__(self):
        check_dataclass_types(self)
        

@dataclass
class MultiPulse:
    """ 
    Constructs/Defines a pulse sequence in the time domain. 

    Attributes:
        gaussian_or_lorentzian: only "gaussian" or "lorentzian" are accepted
        amplitude: the relative scale of the envelopes
        delay: the relative delay (not position) of each pulse, has to be one less than the total number of pulses
        duration: the FWHM of each envelope in the time domain
        central_frequency: the central frequency of each pulse
        p: the super-gaussian/lorentzian of each pulse in the time domain
        phase: the spectral phase of each pulse
        envelope: is constructed from the given attributes
    """

    gaussian_or_lorentzian: Union[list,tuple,np.ndarray,str]
    amplitude: Union[list,tuple,np.ndarray,float,int]
    delay: Union[list,np.ndarray,tuple,float,int]
    duration: Union[list,tuple,np.ndarray,float, int]
    central_frequency: Union[list,tuple,np.ndarray,float,int]
    p: Union[list,tuple,np.ndarray,float,int,None]
    phase: Union[list[SpectralPhase],tuple[SpectralPhase], SpectralPhase]
    envelope: Union[list[TemporalAmplitude],tuple[TemporalAmplitude], TemporalAmplitude] = field(compare=False)

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
            env_list.append(TemporalAmplitude(g_or_l[i],amp[i],dt[i],cf[i],p[i]))

        self.envelope = env_list

        check_dataclass_types(self)





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
        Delta_f (float): the frequency range which is used. Extends from -Delta_f to +Delta_f (so it should actually be named f_max?)

    """
    def __init__(self, N=256, Delta_f=None):
        self.N = N
        self.Delta_f = Delta_f


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
            pulse_t = amp_t*np.exp(1j*2*np.pi*env_t.central_frequency)
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

            if self.Delta_f == None:
                f = amplitude.fwhm + amplitude.central_frequency
                fmax = np.maximum(np.abs(np.min(f)), np.abs(np.max(f)))
                fmin, fmax = -4*fmax, 4*fmax
            else:
                fmin, fmax = -1*self.Delta_f, self.Delta_f

        elif isinstance(parameters, CustomPulse):
            assert self.Delta_f != None, "Please predefine a suitable Delta_f for CustomPulse."
            fmin, fmax = -1*self.Delta_f, self.Delta_f
            parameters = parameters, parameters

        elif isinstance(parameters, MultiPulse):
            if self.Delta_f == None:
                Delta_t = 2*(np.max(parameters.delay) - np.min(parameters.delay))
                df = 1/Delta_t
                self.Delta_f = self.N*df
            else:
                pass
            
            fmin, fmax = -1*self.Delta_f, self.Delta_f

        else:
            raise TypeError(f"The input {parameters} has to be a tuple[SpectralAmplitude, SpectralPhase], CustomPulse or MultiPulse. Not {type(parameters)}.")

        assert type(fmax)==int or type(fmax)==float, "There is a type issue with Delta_f."
        self.frequency = np.linspace(fmin, fmax, self.N)
        self.df = np.mean(np.diff(self.frequency))
        self.time = np.fft.fftshift(np.fft.fftfreq(self.N, self.df))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        return parameters



    def generate_pulse(self, parameters):
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
    




    def plot_pulses(self):
        fig = plt.figure(figsize=(9,4))

        plt.subplot(1,2,1)
        plt.plot(self.pulses.time, np.abs(self.pulses.pulse_t))
        plt.xlabel("Time [fs]")
        plt.ylabel("Amplitude [arb. u.]")


        plt.subplot(1,2,2)
        plt.plot(self.pulses.frequency, np.abs(self.pulses.pulse_f))
        plt.xlabel("Frequency [pHz]")
        plt.ylabel("Amplitude [arb. u.]")
