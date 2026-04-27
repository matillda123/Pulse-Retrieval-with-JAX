import jax 
import jax.numpy as jnp

from equinox import tree_at

from functools import partial as Partial


from pulsedjax.core.base_classes_methods import RetrievePulsesSTREAKING
from pulsedjax.core.base_classic_algorithms import normalize_population
from pulsedjax.utilities import calculate_Z_error, calculate_trace, calculate_trace_error

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
        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t, signal_t_new)

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







































class COPRA(COPRABASESTREAKING, RetrievePulsesSTREAKING):
    __doc__ = COPRABASESTREAKING.__doc__
    
    def __init__(self, delay_fs, energy_eV, measured_trace, df_PHz=0.01, Ip_eV=jnp.array([0]), retrieve_dtme=False, cross_correlation="doubleblind", interferometric=False, **kwargs):
        super().__init__(delay_fs, energy_eV, measured_trace, df_PHz=df_PHz, Ip_eV=Ip_eV, retrieve_dtme=retrieve_dtme, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)



    

    def local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        """ Peforms one local iteration. Calls do_iteration() with the appropriate (randomized) signal fields. """
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                         measurement_info, descent_info, "_local", 
                                                         axes=(0,0,0,None,None,None))
        
        local_state = descent_state._local
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                    descent_state.population, local_state, 
                                                    measurement_info, descent_info, 
                                                   "pulse", "_local")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)


        if measurement_info.doubleblind==True:
            signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
            signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                            measurement_info, descent_info, "_local", 
                                                            axes=(0,0,0,None,None,None))
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                        descent_state.population, local_state, 
                                                        measurement_info, descent_info, 
                                                        "gate", "_local")
            population = normalize_population(population, measurement_info, descent_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)



        if measurement_info.retrieve_dtme==True:
            signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
            signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                            measurement_info, descent_info, "_local", 
                                                            axes=(0,0,0,None,None,None))
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                        descent_state.population, local_state, 
                                                        measurement_info, descent_info, 
                                                        "dtme", "_local")
            descent_state = tree_at(lambda x: x.population.dtme, descent_state, population.dtme)

        descent_state = tree_at(lambda x: x._local, descent_state, local_state)
        return descent_state, None
    






    

    def global_step(self, descent_state, measurement_info, descent_info):
        """
        Performs one global iteration of the Common Pulse Retrieval Algorithm. 
        This means the method updates the population once using all measured data at once.
        
        Args:
            descent_state (Pytree):
            measurement_info (Pytree):
            descent_info (Pytree):

        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current trace errors of the population.
        """

        measured_trace = measurement_info.measured_trace
        _calculate_trace = Partial(calculate_trace, measured_trace=measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                         measurement_info, descent_info, "_global", 
                                                         axes=(0,None,0,None,None,None))

        global_state = descent_state._global
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                     descent_state.population, global_state, measurement_info, 
                                                     descent_info, "pulse", "_global")
        
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)

        if measurement_info.doubleblind==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
            signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                            measurement_info, descent_info, "_global", 
                                                            axes=(0,None,0,None,None,None))
        
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                         descent_state.population, global_state, measurement_info, 
                                                         descent_info, "gate", "_global")
            
            population = normalize_population(population, measurement_info, descent_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)




        if measurement_info.retrieve_dtme==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
            signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                            measurement_info, descent_info, "_global", 
                                                            axes=(0,None,0,None,None,None))
        
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                         descent_state.population, global_state, measurement_info, 
                                                         descent_info, "dtme", "_global")
            descent_state = tree_at(lambda x: x.population.dtme, descent_state, population.dtme)

            
        descent_state = tree_at(lambda x: x._global, descent_state, global_state)

        #signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        #trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu, trace, measured_trace)

        descent_state = tree_at(lambda x: x._global.mu, descent_state, mu)
        return descent_state, trace_error.reshape(-1,1)
    
    






    def get_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        raise NotImplementedError

        # descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
        #                                                                  signal_t.signal_t, signal_t_new, tau_arr, 
        #                                                                  local_or_global_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate)
        # return descent_direction, newton_state




