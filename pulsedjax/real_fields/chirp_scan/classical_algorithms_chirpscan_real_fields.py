import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from equinox import tree_at

from pulsedjax.real_fields.base_classes_methods import RetrievePulsesCHIRPSCANwithRealFields
from pulsedjax.core.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE

from pulsedjax.core.gradients.gradients_via_AD import calc_grad_AD_z_error, calc_grad_AD_pie_error




class GeneralizedProjection(RetrievePulsesCHIRPSCANwithRealFields, GeneralizedProjectionBASE):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None, None), **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, f_range_fields=f_range_fields, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, phase_matrix, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        return calc_grad_AD_z_error(population, phase_matrix, signal_t_new, None, measurement_info, self.calculate_signal_t, pulse_or_gate)


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, phase_matrix, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        raise ValueError("Hessian is to expensive")
    

    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        pulse = individual.pulse + gamma*descent_direction
        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual








class PtychographicIterativeEngine(RetrievePulsesCHIRPSCANwithRealFields, PtychographicIterativeEngineBASE):
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None, None), **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, f_range_fields=f_range_fields, **kwargs)



    def reverse_transform_grad(self, signal, phase_matrix, measurement_info):
        """ For Chirp-Scan the effects of the phase matrix have to be undone to obtain the actual PIE-direction. """
        sk, rn = measurement_info.sk_big, measurement_info.rn_big
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f*jnp.exp(-1j*phase_matrix)
        signal = self.ifft(signal_f, sk, rn)
        return signal



    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, phase_matrix_m, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for a given shift. """
        alpha = descent_info.alpha

        grad = jax.vmap(calc_grad_AD_pie_error, in_axes=(0,0,0,0,None,None,None))(population, phase_matrix_m, signal_t_new, measured_trace, measurement_info, 
                                                                                  self.calculate_signal_t, pulse_or_gate)


        U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(signal_t.gate_disp, alpha, pie_method)

        # reverse transform of U only because grad is with respect to pulse and not Amk. 
        U = jax.vmap(self.reverse_transform_grad, in_axes=(0,0,None))(U, phase_matrix_m, measurement_info)
        U, _ = jax.vmap(self.interpolate_signal, in_axes=(0,None,None,None))(U, measurement_info, "big", "main")
        return grad*U


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn
        
        pulse_t=self.ifft(individual.pulse, sk, rn)
        pulse_t=pulse_t + gamma*descent_direction
        pulse = self.fft(pulse_t, sk, rn)

        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual
    

    def calculate_PIE_newton_direction(self, grad, signal_t, phase_matrix, measured_trace, population, local_or_global_state, measurement_info, 
                                                descent_info, pulse_or_gate, local_or_global):
        raise ValueError("Hessian is to expensive.")















class COPRA(RetrievePulsesCHIRPSCANwithRealFields, COPRABASE):
    __doc__ = COPRABASE.__doc__


    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None, None), **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, f_range_fields=f_range_fields, **kwargs)


    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual based on a desent direction and a step size. """
        pulse = individual.pulse + gamma*descent_direction
        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual


    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, phase_matrix, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        return calc_grad_AD_z_error(population, phase_matrix, signal_t_new, None, measurement_info, self.calculate_signal_t, pulse_or_gate)


    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, phase_matrix, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        raise ValueError("Hessian is to expensive.")
