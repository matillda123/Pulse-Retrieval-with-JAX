import jax.numpy as jnp
import jax
from jax.scipy.special import bernoulli, factorial
from jax.tree_util import Partial

from equinox import tree_at

from BaseClasses import RetrievePulses2DSI, ClassicAlgorithmsBASE
from utilities import MyNamespace, center_signal



class DirectReconstruction(ClassicAlgorithmsBASE, RetrievePulses2DSI):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog, **kwargs)

        self.integration_method = "euler_maclaurin_3"
        self.name = "DirectReconstruction"


    def apply_hann_window(self, signal, axis=-1):
        N=jnp.shape(signal)[axis]
        n=jnp.arange(N)
        hann = jnp.sin(jnp.pi*n/N)**2
        return jnp.swapaxes(jnp.swapaxes(signal, -1, axis)*hann, axis, -1)


    def integrate_signal_1D(self, signal, x, method):
        dx = jnp.mean(jnp.diff(x))

        if method=="cumsum":
            signal = jnp.cumsum(signal, axis=-1)*dx
            
        elif method[:-2]=="euler_maclaurin":
            n = jnp.asarray(int(method[-1]))
            bn = bernoulli(2*n)

            y_prime = jnp.gradient(signal, x, axis=-1)
            t = dx**2/12*(y_prime[:-1] - y_prime[1:])
            for i in jnp.arange(3, 2*n+1, 2):
                f = bn[i+1]/factorial(i+1)
                y_prime = jnp.gradient(jnp.gradient(y_prime, x, axis=-1), x, axis=-1)
                t = t + dx**(i+1)*f*(y_prime[:-1] - y_prime[1:])

            # the addition of t is correct because the gradients are subtracted in reverse
            yint = dx/2*(signal[:-1] + signal[1:]) + t
            yint = jnp.concatenate((jnp.zeros(1), yint), axis=-1)
            signal = jnp.cumsum(yint, axis=-1)

        else:
            print(f"method must be one cumsum or euler_maclaurin_n. not {method}")
        return signal
    

    def reconstruct_2dsi_1dfft(self, descent_state, measurement_info, descent_info):
        frequency, trace = measurement_info.frequency, measurement_info.measured_trace
        pulse_spectral_amplitude, shear_frequency = measurement_info.spectral_amplitude.pulse, measurement_info.shear_frequency

        integration_method = descent_info.integration_method

        trace_hann = self.apply_hann_window(trace, axis=0)
        trace_f = jnp.fft.fftshift(jnp.fft.fft(trace_hann, axis=0), axes=0)
        idx = jnp.shape(trace)[0] - jnp.argmax(jnp.sum(jnp.abs(trace_f), axis=0))

        group_delay = jnp.unwrap(jnp.angle(trace_f[idx]))/shear_frequency
        spectral_phase = self.integrate_signal_1D(group_delay, frequency, integration_method)
        spectral_phase = spectral_phase - jnp.mean(spectral_phase)
        
        pulse_f = pulse_spectral_amplitude*jnp.exp(1j*spectral_phase)
        pulse_t = jnp.fft.ifft(jnp.fft.fftshift(pulse_f))
        pulse_t = center_signal(pulse_t)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, pulse_t)
        return descent_state
    

    def error_reconstruction(self, descent_state, measurement_info, descent_info):
        pass

    

    def initialize_run(self, population):
        self.descent_info = self.descent_info.expand(integration_method = self.integration_method)
        self.descent_state = self.descent_state.expand(population = population)

        do_run = Partial(self.reconstruct_2dsi_1dfft, measurement_info=self.measurement_info, descent_info=self.descent_info)
        return self.descent_state, do_run

        

    def run():
        pass