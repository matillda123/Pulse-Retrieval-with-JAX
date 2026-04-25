import jax 
import jax.numpy as jnp

from equinox import tree_at


from pulsedjax.core.base_classes_methods import RetrievePulsesSTREAKING
from pulsedjax.core.base_classic_algorithms import normalize_population
from pulsedjax.utilities import calculate_Z_error

from pulsedjax.streaking.base_classes_streaking import GeneralizedProjectionBASESTREAKING, PtychographicIterativeEngineBASESTREAKING, COPRABASESTREAKING

from pulsedjax.core.gradients.streaking_z_error_gradients import calculate_Z_gradient
#from pulsedjax.core.hessians.frog_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error



class GeneralizedProjection(GeneralizedProjectionBASESTREAKING, RetrievePulsesSTREAKING):
    __doc__ = GeneralizedProjectionBASESTREAKING.__doc__

    def __init__(self, delay_fs, energy_eV, measured_trace, df_PHz=0.01, Ip_eV=jnp.array([0]), retrieve_dtme=False, cross_correlation="doubleblind", interferometric=False, **kwargs):
        super().__init__(delay_fs, energy_eV, measured_trace, df_PHz=df_PHz, Ip_eV=Ip_eV, retrieve_dtme=retrieve_dtme, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)


    def do_descent_Z_error_step(self, descent_state, signal_t_new, measurement_info, descent_info):
        """ Does one Z-error descent step. Calls descent_Z_error_step for pulse and or gate. """

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)

        population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)
        
        if measurement_info.doubleblind==True:
            population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, 
                                                            measurement_info, descent_info, "gate")
            population = normalize_population(population, measurement_info, descent_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)

        if measurement_info.retrieve_dtme==True:
            population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, 
                                                            measurement_info, descent_info, "dtme")
            descent_state = tree_at(lambda x: x.population.dtme, descent_state, population.dtme)

        return descent_state, None



    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        raise NotImplementedError("Streaking is already super-expensive to retrieve, dont do this.")
    
        # descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
        #                                                                  signal_t.signal_t, signal_t_new, tau_arr, 
        #                                                                  descent_state, measurement_info, descent_info, 
        #                                                                  full_or_diagonal, pulse_or_gate)
        # return descent_direction, newton_state

