import jax.numpy as jnp
import jax
from jax.scipy.special import bernoulli, factorial
from jax.tree_util import Partial

from equinox import tree_at

from BaseClasses import RetrievePulses2DSI, AlgorithmsBASE
from classic_algorithms_base import GeneralizedProjectionBASE, TimeDomainPtychographyBASE, COPRABASE
from utilities import scan_helper, do_fft, do_ifft, MyNamespace, center_signal, do_interpolation_1d, calculate_trace, calculate_trace_error


from TwoDSI_z_error_gradients import calculate_Z_gradient
from TwoDSI_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pie_pseudo_hessian import PIE_get_pseudo_newton_direction



class DirectReconstruction(AlgorithmsBASE, RetrievePulses2DSI):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, **kwargs):
        assert cross_correlation==True, "DirectReconstruction cannot work for Doubleblind or Autocorrelation-like methods"
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, **kwargs)

        self.name = "DirectReconstruction"

        self.integration_method = "euler_maclaurin_3"
        self.integration_order = None

        self.use_windowing = "hamming"
        self.cut_off_intensity_for_GD = 0.0001


    def apply_windowing(self, a0, signal, axis=-1):
        N=jnp.shape(signal)[axis]
        n=jnp.arange(N)
        window = a0 - (1-a0)*jnp.cos(2*jnp.pi*n/N)
        return jnp.swapaxes(jnp.swapaxes(signal, -1, axis)*window, axis, -1)


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
        frequency = measurement_info.frequency

        idx_trace = jnp.argmax(jnp.mean(measurement_info.measured_trace, axis=0))
        idx_spectrum = jnp.argmax(measurement_info.spectral_amplitude.pulse)
        f_shift = frequency[idx_trace] - frequency[idx_spectrum]

        spectral_phase = do_interpolation_1d(frequency, frequency - f_shift, spectral_phase)
        return spectral_phase
    


    def reconstruct_2dsi_1dfft(self, descent_state, measurement_info, descent_info):
        tau_arr, frequency, trace = measurement_info.tau_arr, measurement_info.frequency, measurement_info.measured_trace
        pulse_spectral_amplitude, anc1_frequency, anc2_frequency = measurement_info.spectral_amplitude.pulse, measurement_info.anc1_frequency, measurement_info.anc2_frequency

        use_windowing = descent_info.use_windowing
        if use_windowing!=False:
            trace_wind = self.apply_windowing(use_windowing, trace, axis=0)
        else:
            trace_wind = trace
        trace_f = jnp.fft.fftshift(jnp.fft.fft(trace_wind, axis=0), axes=0)

        frequency_tau = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(tau_arr), jnp.mean(jnp.diff(tau_arr))))
        shear_frequency_mean = (anc1_frequency + anc2_frequency)/2
        idx = jnp.argmin(jnp.abs(frequency_tau - shear_frequency_mean))

        signal_abs = jnp.abs(trace_f)[idx]
        signal_angle = jnp.angle(trace_f)[idx]

        group_delay = jnp.unwrap(signal_angle)/(anc2_frequency - anc1_frequency)
        group_delay = jnp.where(signal_abs < measurement_info.cut_off_intensity*jnp.max(signal_abs), 0, group_delay)
        group_delay = group_delay - jnp.mean(group_delay)

        group_delay = self.interpolate_group_delay_onto_spectral_amplitude(group_delay, measurement_info)
        spectral_phase = self.integrate_signal_1D(group_delay, frequency, descent_info)

        descent_state = tree_at(lambda x: x.group_delay, descent_state, group_delay)
        descent_state = tree_at(lambda x: x.spectral_phase, descent_state, spectral_phase)

        pulse_f = pulse_spectral_amplitude*jnp.exp(1j*spectral_phase)
        pulse_t = do_ifft(pulse_f, measurement_info.sk, measurement_info.rn)
        pulse_t = center_signal(pulse_t).reshape(1,-1)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, pulse_t)
        return descent_state
    
    

    def calc_error_of_reconstruction(self, descent_state, measurement_info, descent_info):
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        trace = calculate_trace(signal_f)
        # if population is larger than one this may cause an error or bug
        trace_error = calculate_trace_error(trace, measurement_info.measured_trace)
        return trace_error
    


    def step(self, descent_state, measurement_info, descent_info):
        descent_state = self.reconstruct_2dsi_1dfft(descent_state, measurement_info, descent_info)
        trace_error = self.calc_error_of_reconstruction(descent_state, measurement_info, descent_info)
        return descent_state, jnp.asarray([trace_error, trace_error])

    

    def initialize_run(self, population):
        assert self.descent_info.measured_spectrum_is_provided.pulse==True, "You need to provide a spectrum for the pulse."
        assert len(population.pulse)==1, "DirectReconstruction has no inherent randomness, so its not sensible to use or expect more than one result."

        self.measurement_info = self.measurement_info.expand(cut_off_intensity = self.cut_off_intensity_for_GD)

        if self.integration_method[:-2]=="euler_maclaurin":
            self.integration_order = int(self.integration_method[-1])
            self.integration_method = "euler_maclaurin"

        a0_dict = {"hamming": 0.54, 
                   "hann": 0.5,
                   False: False}
            
        self.descent_info = self.descent_info.expand(integration_method = self.integration_method, 
                                                     integration_order = self.integration_order,
                                                     use_windowing = a0_dict[self.use_windowing])
        
        init_arr = jnp.zeros(jnp.size(self.measurement_info.frequency))
        self.descent_state = self.descent_state.expand(population = population, 
                                                       group_delay = init_arr, 
                                                       spectral_phase=init_arr)

        do_scan = Partial(self.step, measurement_info=self.measurement_info, descent_info=self.descent_info)
        do_scan=Partial(scan_helper, actual_function=do_scan, number_of_args=1, number_of_xs=0)
        return self.descent_state, do_scan










class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulses2DSI):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, **kwargs)





    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.gate_pulses, signal_t.gate, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, use_hessian, pulse_or_gate):
        descent_direction, hessian = get_pseudo_newton_direction_Z_error(grad, descent_state.population.pulse, signal_t.gate_pulses, signal_t.gate, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         descent_state.hessian, descent_info.hessian, use_hessian, pulse_or_gate)
        return descent_direction, hessian


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = do_fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = do_ifft(pulse_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse)
        return individual









class TimeDomainPtychography(TimeDomainPtychographyBASE, RetrievePulses2DSI):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, pie_method="rPIE", **kwargs):
        assert cross_correlation!="doubleblind", "Doubleblind is not implemented for 2DSI-TimeDomainPtychography."
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, **kwargs)

        self.pie_method=pie_method


    # def reverse_transform_grad(self, signal, tau_arr, measurement_info):
    #     frequency, time = measurement_info.frequency, measurement_info.time

    #     signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(0, 0, None, None, None))
    #     return signal
    
    

    # def modify_grad_for_gate_pulse(self, grad_all_m, gate_pulse_shifted, nonlinear_method):
    #     pass


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        alpha = descent_info.alpha
        difference_signal_t = signal_t_new - signal_t.signal_t

        probe = signal_t.gate
        grad = -1*jnp.conjugate(probe)*difference_signal_t
        U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
        
        return grad, U
    


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        signal = getattr(individual, pulse_or_gate)
        signal = signal + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual


    # def get_gate_probe_for_hessian(self, pulse_t, gate_pulse_shifted, nonlinear_method):
    #     pass


    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        
        newton_direction_prev = getattr(local_or_global_state.hessian, pulse_or_gate).newton_direction_prev
        
        probe = signal_t.gate

        reverse_transform=None
        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction, hessian = PIE_get_pseudo_newton_direction(grad, probe, signal_f, tau_arr, measured_trace, reverse_transform, newton_direction_prev, 
                                                                     measurement_info, descent_info, pulse_or_gate, local_or_global)
        return descent_direction, hessian
    
    










class COPRA(COPRABASE, RetrievePulses2DSI):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation, anc1_frequency, anc2_frequency, **kwargs)

    

    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = do_fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = do_ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual



    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.gate_pulses, signal_t.gate, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           use_hessian, pulse_or_gate):
        
        hessian_state = local_or_global_state.hessian
        descent_direction, hessian = get_pseudo_newton_direction_Z_error(grad, population.pulse, signal_t.gate_pulses, signal_t.gate, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         hessian_state, descent_info.hessian, use_hessian, pulse_or_gate)
        return descent_direction, hessian