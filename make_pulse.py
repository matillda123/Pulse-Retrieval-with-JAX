from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d

import jax
from utilities import get_sk_rn, do_fft, do_ifft, MyNamespace, do_interpolation_1d, generate_random_continuous_function




@dataclass
class GaussianAmplitude:
    amplitude: np.ndarray 
    central_frequency: np.ndarray 
    fwhm: np.ndarray 


@dataclass
class PolynomialPhase:
    central_frequency: np.ndarray 
    coefficients: np.ndarray 


@dataclass
class SinusoidalPhase:
    amplitude: np.ndarray 
    periodicity: np.ndarray 
    phase_shift: np.ndarray 


@dataclass
class RandomPhase:
    number_of_points: int = 10
    minval_phase: float = -4*np.pi
    maxval_phase: float = 4*np.pi
    sigma: float = 10.0

@dataclass
class CustomPulse:
    frequency: np.ndarray 
    amplitude: np.ndarray 
    phase: np.ndarray 


@dataclass
class MultiPulse:
   delay: np.ndarray 
   duration: np.ndarray 
   central_frequency: np.ndarray 
   amplitude: np.ndarray 
   phase: list 





class MakePulse:
    """ 
    Is supposed to generate pulses based on input parameters from GaussianAmplitude, PolynomialPhase, SinusoidalPhase, RandomPhase, CustomPulse or MultiPulse.
    Does not work with Jax.

    Attributes:
        N: int,
        Delta_f: float,

    """
    def __init__(self, N=256, Delta_f=None):
        self.N = N
        self.Delta_f = Delta_f


    def gaussian(self, x, amp, fwhm, shift):
        b=fwhm/2.355 # fwhm to sigma
        idx = np.argmin(np.abs(x-shift))
        shift_new = x[idx] # makes sure that shift lies on top of grid point, prevents issues when fwhm is to small
        return amp*np.exp(-(x-shift_new)**2/(2*b**2))
    

    def generate_gaussian_amplitude(self, x, amp, fwhm, shift):
        assert len(amp)==len(fwhm)==len(shift)

        a = 0
        for i in range(len(amp)):
            a = a + self.gaussian(x, amp[i], fwhm[i], shift[i])
        return a
    

    def generate_polynomial_phase(self, frequency, central_frequency, parameters):
        n = len(parameters)
        phase = 0
        for i in range(n):
            phase = phase + parameters[i]*(frequency-central_frequency)**i
        return 2*np.pi*phase
    
    
    def generate_sinusoidal_phase(self, frequency, a, b, c):
        assert len(a)==len(b)==len(c)

        phase = 0
        for i in range(len(a)):
            phase = phase + a[i]*np.sin(2*np.pi*b[i]*frequency-c[i])
        return 2*np.pi*phase
    

    def get_spectral_phase(self, phase_parameters, amp_f=None):
        if isinstance(phase_parameters, PolynomialPhase):
            central_frequency, coefficients = phase_parameters.central_frequency, phase_parameters.coefficients
            phase = self.generate_polynomial_phase(self.frequency, central_frequency, coefficients)

        elif isinstance(phase_parameters, SinusoidalPhase):
            a,b,c = phase_parameters.amplitude, phase_parameters.periodicity, phase_parameters.phase_shift
            phase = self.generate_sinusoidal_phase(self.frequency, a, b, c)

        elif isinstance(phase_parameters, RandomPhase):
            N_points, sigma = phase_parameters.number_of_points, phase_parameters.sigma
            min_val, max_val = phase_parameters.minval_phase, phase_parameters.maxval_phase
            phase = generate_random_continuous_function(jax.random.PRNGKey(np.random.randint(0,1e9)), N_points, self.frequency, 
                                                        min_val, max_val, gaussian_filter1d(amp_f, sigma=sigma))
            
        elif isinstance(phase_parameters, CustomPulse):
            phase = do_interpolation_1d(self.frequency, phase_parameters.frequency, phase_parameters.phase)

        else:
            raise NotImplementedError(f"Phase type {phase_parameters} is not implemented.")

        return phase
    


    def get_spectral_amplitude(self, amp_parameters):
        if isinstance(amp_parameters, GaussianAmplitude):
            amp, fwhm, shift = amp_parameters.amplitude, amp_parameters.fwhm, amp_parameters.central_frequency
            amp_f = self.generate_gaussian_amplitude(self.frequency, amp, fwhm, shift)

        elif isinstance(amp_parameters, MultiPulse):
            amp, fwhm, shift = amp_parameters.amplitude, amp_parameters.duration, amp_parameters.delay
            shift = np.concatenate((np.zeros(1), shift))
            shift = shift - np.mean(shift)
            amp_t = self.generate_gaussian_amplitude(self.time, amp, fwhm, shift)
            amp_f = do_fft(amp_t, self.sk, self.rn)
            amp_f = np.abs(amp_f)
            
        else:
            raise NotImplementedError(f"Amplitude type {amp_parameters} is not implemented.")

        return amp_f
    
        
        
    def init_generation(self, parameters):
        if isinstance(parameters, tuple)==True:
            amplitude, phase = parameters

            if self.Delta_f == None:
                f = amplitude.fwhm + amplitude.central_frequency
                fmax = np.maximum(np.abs(np.min(f)), np.abs(np.max(f)))
                fmin, fmax = -4*fmax, 4*fmax
            else:
                fmin, fmax = -1*self.Delta_f, self.Delta_f

        else:
            assert self.Delta_f != None
            fmin, fmax = -1*self.Delta_f, self.Delta_f

            amplitude = parameters
            if isinstance(parameters, MultiPulse):
                phase = parameters.phase # this could be a list of phase_classes
            else:
                phase = parameters


        self.frequency = np.linspace(fmin, fmax, self.N)
        self.df = np.mean(np.diff(self.frequency))
        self.time = np.fft.fftshift(np.fft.fftfreq(self.N, self.df))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        return amplitude, phase



    def generate_pulse(self, parameters):
        amplitude, phase = self.init_generation(parameters)

        amp_f = self.get_spectral_amplitude(amplitude)

        if isinstance(parameters, MultiPulse): 
            shift = np.concatenate((np.zeros(1), parameters.delay))
            shift = shift - np.mean(shift)
        
            temp = 0
            for i in range(len(shift)):
                amp_t = self.gaussian(self.time, parameters.amplitude[i], parameters.duration[i], shift[i])
                pulse_t = amp_t * np.exp(1j*2*np.pi*parameters.central_frequency[i]*self.time)
                pulse_f = do_fft(pulse_t, self.sk, self.rn)

                phase = self.get_spectral_phase(parameters.phase[i], amp_f=np.abs(pulse_f))
                pulse_f = pulse_f*np.exp(1j*phase)

                temp = temp + pulse_f
            pulse_f = temp

        else:
            phase = self.get_spectral_phase(phase, amp_f=amp_f)
            pulse_f = amp_f*np.exp(1j*phase)

        pulse_t = do_ifft(pulse_f, self.sk, self.rn)

        self.pulses = MyNamespace(time=self.time, frequency=self.frequency, pulse_t=pulse_t, pulse_f=pulse_f)
        return self.time, pulse_t, self.frequency, pulse_f
