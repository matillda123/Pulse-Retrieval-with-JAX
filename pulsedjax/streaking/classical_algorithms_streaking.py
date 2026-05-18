import jax.numpy as jnp

from pulsedjax.core.base_classes_methods import RetrievePulsesSTREAKING
from pulsedjax.streaking.base_classes_streaking import GeneralizedProjectionBASESTREAKING, PtychographicIterativeEngineBASESTREAKING, COPRABASESTREAKING

from pulsedjax.core.gradients.streaking_z_error_gradients import calculate_Z_gradient
#from pulsedjax.core.hessians.frog_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error



class GeneralizedProjection(GeneralizedProjectionBASESTREAKING, RetrievePulsesSTREAKING):
    __doc__ = GeneralizedProjectionBASESTREAKING.__doc__

    def __init__(self, delay_fs, energy_eV, measured_trace, Ip_eV=jnp.array([0]), retrieve_dtme=False, cross_correlation="doubleblind", interferometric=False, **kwargs):
        super().__init__(delay_fs, energy_eV, measured_trace, Ip_eV=Ip_eV, retrieve_dtme=retrieve_dtme, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)



    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        raise NotImplementedError("Streaking is already super-expensive to retrieve, dont do this.")









class COPRA(COPRABASESTREAKING, RetrievePulsesSTREAKING):
    __doc__ = COPRABASESTREAKING.__doc__
    
    def __init__(self, delay_fs, energy_eV, measured_trace, Ip_eV=jnp.array([0]), retrieve_dtme=False, cross_correlation="doubleblind", interferometric=False, **kwargs):
        super().__init__(delay_fs, energy_eV, measured_trace, Ip_eV=Ip_eV, retrieve_dtme=retrieve_dtme, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)



    def get_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        raise NotImplementedError
