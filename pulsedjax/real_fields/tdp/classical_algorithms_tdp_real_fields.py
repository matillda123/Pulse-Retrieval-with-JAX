from pulsedjax.real_fields.core.base_classes_methods import RetrievePulsesTDPwithRealFields
from pulsedjax.tdp import (LSGPA as _LSGPA, CPCGPA as _CPCGPA,
                            GeneralizedProjection as _GeneralizedProjection, 
                            PtychographicIterativeEngine as _PtychographicIterativeEngine, 
                            COPRA as _COPRA, LSF as _LSF)

from pulsedjax.utilities import MyNamespace, calculate_gate_with_Real_Fields, do_interpolation_1d

from equinox import tree_at
import jax.numpy as jnp
import jax

from functools import partial as Partial


class LSGPA(RetrievePulsesTDPwithRealFields, _LSGPA):
    __doc__ = _LSGPA.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)
        

    def update_pulse(self, pulse, signal_t_new, gate_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the pulse. """
        pulse = super().update_pulse(pulse, signal_t_new, gate_shifted, measurement_info, descent_info)
        pulse, _ = self.interpolate_signal_t(pulse, measurement_info, "big", "main")
        return pulse
    
    
    def update_gate(self, gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the gate. """
        gate = super().update_pulse(gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info)
        gate, _ = self.interpolate_signal_t(gate, measurement_info, "big", "main")
        return gate
    



class CPCGPA(RetrievePulsesTDPwithRealFields, _CPCGPA):
    __doc__ = _CPCGPA.__doc__

    def __init__(self, delay, frequency, trace, nonlinear_method, spectral_filter, cross_correlation=False, constraints=False, svd=False, antialias=False, **kwargs):
        super().__init__(delay, frequency, trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, constraints=constraints, svd=svd, antialias=antialias, **kwargs)

        self.idx_arr = jnp.arange(jnp.size(self.frequency_big)) 
        self.measurement_info = tree_at(lambda x: x.idx_arr, self.measurement_info, self.idx_arr)

        tau_arr_new = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency_big), jnp.mean(jnp.diff(self.frequency_big))))
        self.measured_trace = jax.vmap(Partial(do_interpolation_1d, method="linear"), 
                                       in_axes=(None, None, 1), out_axes=1)(self.tau_arr_new, self.tau_arr, self.measured_trace)
        
        #self.measured_trace = 0.5*(self.measured_trace + jnp.flip(self.measured_trace, axis=1))
        self.measurement_info = tree_at(lambda x: x.tau_arr, self.measurement_info, tau_arr_new)
        self.measurement_info = tree_at(lambda x: x.x_arr, self.measurement_info, tau_arr_new)
        self.measurement_info = tree_at(lambda x: x.transform_arr, self.measurement_info, tau_arr_new)
        self.measurement_info = tree_at(lambda x: x.measured_trace, self.measurement_info, self.measured_trace)

    
    
    def calculate_gate(self, gate_pulse, measurement_info):
        gate_pulse = self.apply_spectral_filter(gate_pulse, measurement_info.spectral_filter, 
                                                measurement_info.sk_big, measurement_info.rn_big)
        return calculate_gate_with_Real_Fields(gate_pulse, measurement_info.nonlinear_method)
    
    
    def calculate_signal_t_using_opf(self, individual, iteration, measurement_info, descent_info):
        """ Calculates signal_t for and individual via the opf. """
        
        pulse_t, pulse_t_prime = individual.pulse, individual.pulse_prime
        pulse_t, _ = self.interpolate_signal_t(pulse_t, measurement_info, "main", "big")
        pulse_t_prime, _ = self.interpolate_signal_t(pulse_t_prime, measurement_info, "main", "big")
        pulse_t, pulse_t_prime = jnp.real(pulse_t), jnp.real(pulse_t_prime)

        if measurement_info.doubleblind==True:
            gate, gate_prime = individual.gate, individual.gate_prime
            gate, _ = self.interpolate_signal_t(gate, measurement_info, "main", "big")
            gate_prime, _ = self.interpolate_signal_t(gate_prime, measurement_info, "main", "big")
            gate, gate_prime = jnp.real(gate), jnp.real(gate_prime)

        elif measurement_info.cross_correlation==True:
            gate = gate_prime = self.calculate_gate(jnp.real(measurement_info.gate), measurement_info)

        else:
            gate = self.calculate_gate(pulse_t, measurement_info)
            gate_prime = self.calculate_gate(pulse_t_prime, measurement_info)

        
        opf = self.calculate_opf(pulse_t, gate, pulse_t_prime, gate_prime, iteration, measurement_info.nonlinear_method, measurement_info)

        if descent_info.antialias==True:
            half_N = jnp.size(opf[0])//2
            opf = self.do_anti_alias(opf, half_N)

        signal_t = self.convert_opf_to_signal_t(opf, measurement_info.idx_arr)
        signal_t = jnp.transpose(signal_t) # transpose for consistency
        signal_f = self.fft(signal_t, measurement_info.sk_big, measurement_info.rn_big)
        mask = measurement_info.mask
        #mask = 0.5*(mask + jnp.flip(mask))
        signal_f = signal_f*mask
        signal_t = self.ifft(signal_f, measurement_info.sk_big, measurement_info.rn_big)
        return MyNamespace(signal_t=signal_t, signal_f=signal_f)



    def update_individual(self, opf, individual, measurement_info, descent_info):
        """ Updates and individual using an updated opf. """
        individual = jax.tree.map(lambda x: jnp.real(self.interpolate_signal_t(x, measurement_info, "main", "big")[0]), individual)
        individual = super().update_individual(opf, individual, measurement_info, descent_info)
        return jax.tree.map(lambda x: self.interpolate_signal_t(x, measurement_info, "big", "main")[0], individual)
    




class GeneralizedProjection(RetrievePulsesTDPwithRealFields, _GeneralizedProjection):
    __doc__ = _GeneralizedProjection.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)



    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """

        pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
        population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))
        # if measurement_info.doubleblind==True:
        #     gate, _ = self.interpolate_signal_t(population.gate, measurement_info, "main", "big")
        #     population = tree_at(lambda x: x.gate, population, jnp.real(gate))

        grad = super().calculate_Z_gradient_individual(signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate)
        return self.interpolate_signal_f(grad, measurement_info, "big", "main")


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        pulse, _ = self.interpolate_signal_t(descent_state.population.pulse, measurement_info, "main", "big")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, jnp.real(pulse))

        descent_direction, newton_state = super().calculate_Z_newton_direction(grad, signal_t_new, signal_t, tau_arr, 
                                                                               descent_state, measurement_info, descent_info, 
                                                                               full_or_diagonal, pulse_or_gate)
        
        descent_direction = self.interpolate_signal_f(descent_direction, measurement_info, "big", "main")
        return descent_direction, newton_state





class PtychographicIterativeEngine(RetrievePulsesTDPwithRealFields, _PtychographicIterativeEngine):
    __doc__ = _PtychographicIterativeEngine.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for pulse or gate-pulse for a given shift. """

        if pulse_or_gate=="gate":
            pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
            population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))

        grad_U = super().calculate_PIE_descent_direction_m(signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate)
        grad_U, _ = self.interpolate_signal_t(grad_U, measurement_info, "big", "main")
        return grad_U
    

    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        """ Calculates the newton direction for a population. """

        if pulse_or_gate=="gate":
            pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
            population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))

        descent_direction, newton_state = super().calculate_PIE_newton_direction(grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, pulse_or_gate, local_or_global)
        descent_direction, _ = self.interpolate_signal_t(descent_direction, measurement_info, "big", "main")
        return descent_direction, newton_state





class COPRA(RetrievePulsesTDPwithRealFields, _COPRA):
    __doc__ = _COPRA.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)


    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """

        pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
        population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))
            
        grad = super().get_Z_gradient_individual(signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate)
        return self.interpolate_signal_f(grad, measurement_info, "big", "main")



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        # the hessian will be inverted on frequency_big 

        pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
        population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))

        descent_direction, newton_state = super().calculate_Z_newton_direction(grad, signal_t, signal_t_new, tau_arr, 
                                                                               population, local_or_global_state, 
                                                                               measurement_info, descent_info, 
                                                                               full_or_diagonal, pulse_or_gate)
        
        descent_direction = self.interpolate_signal_f(descent_direction, measurement_info, "big", "main")
        return descent_direction, newton_state
    










class LSF(RetrievePulsesTDPwithRealFields, _LSF):
    __doc__ = _LSF.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)
        self._post_init()