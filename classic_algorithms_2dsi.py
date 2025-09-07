import jax.numpy as jnp
import jax
from jax.scipy.special import bernoulli, factorial
from jax.tree_util import Partial

from equinox import tree_at

from BaseClasses import RetrievePulses2DSI, AlgorithmsBASE
from utilities import scan_helper, do_fft, do_ifft, MyNamespace, center_signal_to_max, do_interpolation_1d, calculate_trace, calculate_trace_error



class DirectReconstruction(AlgorithmsBASE, RetrievePulses2DSI):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog, anc1_frequency, anc2_frequency, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog, anc1_frequency, anc2_frequency, **kwargs)

        self.integration_method = "euler_maclaurin_3"
        self.integration_order = None
        self.name = "DirectReconstruction"

        assert nonlinear_method=="shg", f"DirectReconstruction only works for three-wave mixing. nonlinear_method cannot be {nonlinear_method}"


    def apply_hann_window(self, signal, axis=-1):
        N=jnp.shape(signal)[axis]
        n=jnp.arange(N)
        hann = jnp.sin(jnp.pi*n/N)**2
        return jnp.swapaxes(jnp.swapaxes(signal, -1, axis)*hann, axis, -1)


    def integrate_signal_1D(self, signal, x, descent_info):
        method, order = descent_info.integration_method, descent_info.integration_order

        dx = jnp.mean(jnp.diff(x))

        if method=="cumsum":
            signal = jnp.cumsum(signal, axis=-1)*dx
            
        elif method=="euler_maclaurin":
            n = order
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
            print(f"method must be one of cumsum or euler_maclaurin. not {method}")
        return signal
    

    def interpolate_group_delay_onto_spectral_amplitude(self, spectral_phase, measurement_info):
        frequency, nonlinear_method = measurement_info.frequency, measurement_info.nonlinear_method

        if nonlinear_method=="shg":
            factor=2
        elif nonlinear_method=="thg":
            factor=3
        else:
            factor=1

        anc1_f, anc2_f = measurement_info.anc1_frequency, measurement_info.anc2_frequency
        f_shift = (anc2_f + anc1_f)/2*(factor-1) + (anc2_f - anc1_f)/2

        spectral_phase = do_interpolation_1d(frequency, frequency - f_shift, spectral_phase)
        return spectral_phase
    


    def reconstruct_2dsi_1dfft(self, descent_state, measurement_info, descent_info):
        frequency, trace = measurement_info.frequency, measurement_info.measured_trace
        pulse_spectral_amplitude, anc1_frequency, anc2_frequency = measurement_info.spectral_amplitude.pulse, measurement_info.anc1_frequency, measurement_info.anc2_frequency

        trace_hann = trace# self.apply_hann_window(trace, axis=0)
        trace_f = jnp.fft.fftshift(jnp.fft.fft(trace_hann, axis=0), axes=0)
        idx = jnp.argsort(jnp.sum(jnp.abs(trace_f), axis=1))[-2]

        signal_abs = jnp.abs(trace_f)[idx]
        signal_angle = jnp.angle(trace_f)[idx]

        group_delay = jnp.unwrap(signal_angle)/(anc2_frequency - anc1_frequency)
        group_delay = jnp.where(signal_abs < 0.0001*jnp.max(signal_abs), 0, group_delay)

        group_delay = self.interpolate_group_delay_onto_spectral_amplitude(group_delay, measurement_info)

        spectral_phase = self.integrate_signal_1D(group_delay, frequency, descent_info)

        descent_state = tree_at(lambda x: x.group_delay, descent_state, group_delay)
        descent_state = tree_at(lambda x: x.spectral_phase, descent_state, spectral_phase)

        pulse_f = pulse_spectral_amplitude*jnp.exp(1j*spectral_phase)
        pulse_t = do_ifft(pulse_f, measurement_info.sk, measurement_info.rn)
        pulse_t = center_signal_to_max(pulse_t).reshape(1,-1)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, pulse_t)
        return descent_state
    
    

    def calc_error_of_reconstruction(self, descent_state, measurement_info, descent_info):
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        trace = calculate_trace(signal_f)
        trace_error = calculate_trace_error(trace, measurement_info.measured_trace)
        return trace_error
    


    def step(self, descent_state, measurement_info, descent_info):
        descent_state = self.reconstruct_2dsi_1dfft(descent_state, measurement_info, descent_info)
        trace_error = self.calc_error_of_reconstruction(descent_state, measurement_info, descent_info)
        return descent_state, jnp.asarray([trace_error, trace_error])

    

    def initialize_run(self, population):
        assert self.descent_info.measured_spectrum_is_provided.pulse==True, "You need to provide a spectrum for the pulse."
        assert len(population.pulse)==1, "DirectReconstruction has no inherent randomness, so its not sensible to use or expect more than one result."

        if self.integration_method[:-2]=="euler_maclaurin":
            self.integration_order = int(self.integration_method[-1])
            self.integration_method = "euler_maclaurin"
            
        self.descent_info = self.descent_info.expand(integration_method = self.integration_method, 
                                                     integration_order=self.integration_order)
        init_arr = jnp.zeros(jnp.size(self.measurement_info.frequency))
        self.descent_state = self.descent_state.expand(population = population, 
                                                       group_delay = init_arr, 
                                                       spectral_phase=init_arr)

        do_scan = Partial(self.step, measurement_info=self.measurement_info, descent_info=self.descent_info)
        do_scan=Partial(scan_helper, actual_function=do_scan, number_of_args=1, number_of_xs=0)
        return self.descent_state, do_scan