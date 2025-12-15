import jax
import jax.numpy as jnp

from equinox import tree_at

from src.real_fields.base_classes_methods import RetrievePulsesFROGwithRealFields
from src.core.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE

from src.core.gradients.gradients_via_AD import calc_grad_AD_z_error, calc_grad_AD_pie_error










class GeneralizedProjection(RetrievePulsesFROGwithRealFields, GeneralizedProjectionBASE):
    """
    The Generalized Projection Algorithm for FROG.
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        return calc_grad_AD_z_error(population, tau_arr, signal_t_new, None, measurement_info, self.calculate_signal_t, pulse_or_gate)


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        raise NotImplementedError("The Z-error hessian could be calculated via AD. But thats very expensive.")
    


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent_direction and step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = self.fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = self.ifft(pulse_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse)
        return individual
















class PtychographicIterativeEngine(RetrievePulsesFROGwithRealFields, PtychographicIterativeEngineBASE):
    """
    The Ptychographic Iterative Engine (PIE) for FROG.

    Attributes:
        pie_method (None, str): specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE. Where None indicates that the pure gradient is used.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, pie_method="rPIE", cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "Dont use interferometric with PIE. its not meant or made for that"

        self.pie_method=pie_method




    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for pulse or gate-pulse for a given shift. """

        grad = jax.vmap(calc_grad_AD_pie_error, in_axes=(0,0,0,0,None,None,None))(population, tau, signal_t_new, 
                                                                                  jnp.abs(signal_t.signal_f), 
                                                                                  measurement_info, self.calculate_signal_t, 
                                                                                  pulse_or_gate)
        alpha = descent_info.alpha
        if pulse_or_gate=="pulse":
            probe = signal_t.gate_shifted
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
            
        elif pulse_or_gate=="gate":
            probe = jnp.broadcast_to(population.pulse, jnp.shape(signal_t.gate_shifted))
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
            U = self.reverse_transform_grad(U, tau, measurement_info)

        return grad, U



    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        raise NotImplementedError("the hessian could be obtained via AD. But thats very expensive.")
    


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and step size. """
        signal = getattr(individual, pulse_or_gate)
        signal = signal + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual
        
    





class COPRA(RetrievePulsesFROGwithRealFields, COPRABASE):
    """
    The Common Pulse Retrieval Algorithm (COPRA) for FROG.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)



    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        return calc_grad_AD_z_error(population, tau_arr, signal_t_new, None, measurement_info, self.calculate_signal_t, pulse_or_gate)



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        raise NotImplementedError(f"The z-error hessian could be calculated via AD. But thats very expensive.")
    



    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = self.ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual

