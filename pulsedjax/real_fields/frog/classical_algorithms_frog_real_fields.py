import jax
import jax.numpy as jnp

from equinox import tree_at

from pulsedjax.real_fields.base_classes_methods import RetrievePulsesFROGwithRealFields
from pulsedjax.core.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE

from pulsedjax.core.gradients.gradients_via_AD import calc_grad_AD_z_error, calc_grad_AD_pie_error










class GeneralizedProjection(RetrievePulsesFROGwithRealFields, GeneralizedProjectionBASE):
    __doc__ = GeneralizedProjectionBASE.__doc__


    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)
        
        not_working=True
        assert not_working==False, "This is running. But not converging."

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
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)

        # interferometiric should work here since the gradient is obtained via AD
        # assert self.interferometric==False, "Dont use interferometric=True with PIE. its not meant or made for that"



    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for pulse or gate-pulse for a given shift. """

        grad = jax.vmap(calc_grad_AD_pie_error, in_axes=(0,0,0,0,None,None,None))(population, tau, signal_t_new, 
                                                                                  measured_trace, 
                                                                                  measurement_info, self.calculate_signal_t, 
                                                                                  pulse_or_gate)
        alpha = descent_info.alpha
        if pulse_or_gate=="pulse":
            probe, _ = jax.vmap(self.interpolate_signal, in_axes=(0,None,None,None))(signal_t.gate_shifted, measurement_info, "big", "main")
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
            
        elif pulse_or_gate=="gate":
            probe = jnp.broadcast_to(population.pulse, jnp.shape(signal_t.gate_shifted)[:2] + (jnp.shape(population.pulse)[-1], ))
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
            # only reverse_transfor U, grad is with respect to pulse and not Amk
            U = jax.vmap(self.reverse_transform_grad, in_axes=(0,0,None))(U, tau, measurement_info)

        return grad*U



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
    __doc__ = COPRABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)

        not_working=True
        assert not_working==False, "This is running. But not converging."


    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calc_grad_AD_z_error(population, tau_arr, signal_t_new, None, measurement_info, self.calculate_signal_t, pulse_or_gate)
        return grad


    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        raise NotImplementedError(f"The z-error hessian could be calculated via AD. But thats very expensive.")
    



    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn
        # this sk, rn should be correct, ad grad is with respect to pulse/gate -> these are defined on main

        signal = getattr(individual, pulse_or_gate)
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = self.ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual

