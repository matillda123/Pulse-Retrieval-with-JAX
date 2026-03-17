from pulsedjax.real_fields.core.base_classes_methods import RetrievePulses2DSIwithRealFields
from pulsedjax.twodsi import (GeneralizedProjection as _GeneralizedProjection, 
                            PtychographicIterativeEngine as _PtychographicIterativeEngine, 
                            COPRA as _COPRA)

from equinox import tree_at
import jax.numpy as jnp



class GeneralizedProjection(RetrievePulses2DSIwithRealFields, _GeneralizedProjection):
    __doc__ = _GeneralizedProjection.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)
     
    
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
        # if measurement_info.doubleblind==True:
        #     gate, _ = self.interpolate_signal_t(descent_state.population.gate, measurement_info, "main", "big")
        #     descent_state = tree_at(lambda x: x.descent_state.gate, descent_state, jnp.real(gate))

        descent_direction, newton_state = super().calculate_Z_newton_direction(grad, signal_t_new, signal_t, tau_arr, 
                                                                               descent_state, measurement_info, descent_info, 
                                                                               full_or_diagonal, pulse_or_gate)
        
        descent_direction = self.interpolate_signal_f(descent_direction, measurement_info, "big", "main")
        return descent_direction, newton_state
    




class PtychographicIterativeEngine(RetrievePulses2DSIwithRealFields, _PtychographicIterativeEngine):
    __doc__ = _PtychographicIterativeEngine.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs): 
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for pulse or gate-pulse for a given shift. """

        if pulse_or_gate=="gate":
            pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
            population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))
        # if measurement_info.doubleblind==True:
        #     gate, _ = self.interpolate_signal_t(population.gate, measurement_info, "main", "big")
        #     population = tree_at(lambda x: x.gate, population, jnp.real(gate))

        grad_U = super().calculate_PIE_descent_direction_m(signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate)
        grad_U, _ = self.interpolate_signal_t(grad_U, measurement_info, "big", "main")
        return grad_U
    

    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        """ Calculates the newton direction for a population. """

        if pulse_or_gate=="gate":
            pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
            population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))
        # if measurement_info.doubleblind==True:
        #     gate, _ = self.interpolate_signal_t(population.gate, measurement_info, "main", "big")
        #     population = tree_at(lambda x: x.gate, population, jnp.real(gate))

        descent_direction, newton_state = super().calculate_PIE_newton_direction(grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, pulse_or_gate, local_or_global)
        descent_direction, _ = self.interpolate_signal_t(descent_direction, measurement_info, "big", "main")
        return descent_direction, newton_state


        

class COPRA(RetrievePulses2DSIwithRealFields, _COPRA):
    __doc__ = _COPRA.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, **kwargs)
    
    
    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """

        pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
        population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))
        # if measurement_info.doubleblind==True:
        #     gate, _ = self.interpolate_signal_t(population.gate, measurement_info, "main", "big")
        #     population = tree_at(lambda x: x.gate, population, jnp.real(gate))
            
        grad = super().get_Z_gradient_individual(signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate)
        return self.interpolate_signal_f(grad, measurement_info, "big", "main")



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        # the hessian will be inverted on frequency_big 

        pulse, _ = self.interpolate_signal_t(population.pulse, measurement_info, "main", "big")
        population = tree_at(lambda x: x.pulse, population, jnp.real(pulse))
        # if measurement_info.doubleblind==True:
        #     gate, _ = self.interpolate_signal_t(population.gate, measurement_info, "main", "big")
        #     population = tree_at(lambda x: x.gate, population, jnp.real(gate))

        descent_direction, newton_state = super().calculate_Z_newton_direction(grad, signal_t, signal_t_new, tau_arr, 
                                                                               population, local_or_global_state, 
                                                                               measurement_info, descent_info, 
                                                                               full_or_diagonal, pulse_or_gate)
        
        descent_direction = self.interpolate_signal_f(descent_direction, measurement_info, "big", "main")
        return descent_direction, newton_state
    