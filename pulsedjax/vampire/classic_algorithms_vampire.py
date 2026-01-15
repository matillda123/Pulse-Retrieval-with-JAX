import jax.numpy as jnp
import jax

from equinox import tree_at

import refractiveindex

from pulsedjax.core.base_classes_methods import RetrievePulsesVAMPIRE
from pulsedjax.core.base_classic_algorithms import LSGPABASE, CPCGPABASE, GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE

from pulsedjax.core.gradients.vampire_z_error_gradients import calculate_Z_gradient
from pulsedjax.core.hessians.vampire_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pulsedjax.core.hessians.pie_pseudo_hessian import PIE_get_pseudo_newton_direction

from pulsedjax.utilities import calculate_gate



class LSGPA(LSGPABASE, RetrievePulsesVAMPIRE):
    __doc__ = LSGPABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, **kwargs)



class CPCGPA(CPCGPABASE, RetrievePulsesVAMPIRE):
    __doc__ = CPCGPABASE.__doc__

    def __init__(self, delay, frequency, trace, nonlinear_method, spectral_filter, cross_correlation=False, constraints=False, svd=False, antialias=False, **kwargs):
        super().__init__(delay, frequency, trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, constraints=constraints, svd=svd, antialias=antialias, **kwargs)

    
    def calculate_gate(self, gate_pulse, measurement_info):
        tau, nonlinear_method = measurement_info.tau_interferometer, measurement_info.nonlinear_method
        sk, rn, frequency, time = measurement_info.sk, measurement_info.rn, measurement_info.frequency, measurement_info.time
        gate_disp = self.apply_phase(gate_pulse, measurement_info, sk, rn) 
        gate_pulse = self.calculate_shifted_signal(gate_pulse, frequency, jnp.asarray([tau]), time)
        gate_pulses = jnp.squeeze(gate_pulse) + gate_disp
        return calculate_gate(gate_pulses, nonlinear_method)







class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulsesVAMPIRE):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.gate_pulses, signal_t.gate, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, descent_state.population.pulse, signal_t.gate_pulses, signal_t.gate, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         descent_state.newton, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and step size."""
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = self.fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = self.ifft(pulse_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse)
        return individual









class PtychographicIterativeEngine(PtychographicIterativeEngineBASE, RetrievePulsesVAMPIRE):
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, **kwargs):
        assert cross_correlation!="doubleblind", "Doubleblind is not implemented for VAMPIRE-PtychographicIterativeEngine."
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, **kwargs)


    # def reverse_transform_grad(self, signal, tau_arr, measurement_info):
    #     frequency, time = measurement_info.frequency, measurement_info.time
    #     signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(0, 0, None, None, None))
    #     return signal

    # def modify_grad_for_gate_pulse(self, grad_all_m, gate_pulse_shifted, nonlinear_method):
    #     pass


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for a given shift. """
        alpha = descent_info.alpha
        difference_signal_t = signal_t_new - signal_t.signal_t

        probe = signal_t.gate
        grad = -1*jnp.conjugate(probe)*difference_signal_t
        U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
        
        return grad*U
    

    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and step size. """
        signal = getattr(individual, pulse_or_gate)
        signal = signal + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual


    # def get_gate_probe_for_hessian(self, pulse_t, gate_pulse_shifted, nonlinear_method):
    #     pass


    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        
        """ Calculates the PIE newton direction for a population. """
        
        newton_direction_prev = getattr(local_or_global_state.newton, pulse_or_gate).newton_direction_prev
        probe = signal_t.gate

        reverse_transform = None
        #signal_f = self.fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction, newton_state = PIE_get_pseudo_newton_direction(grad, probe, signal_t.signal_f, tau_arr, measured_trace, reverse_transform, newton_direction_prev, 
                                                                     measurement_info, descent_info, pulse_or_gate, local_or_global)
        return descent_direction, newton_state
    
    










class COPRA(COPRABASE, RetrievePulsesVAMPIRE):
    __doc__ = COPRABASE.__doc__
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, **kwargs)


    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual via a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = self.ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual



    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.gate_pulses, signal_t.gate, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        
        newton_state = local_or_global_state.newton
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, population.pulse, signal_t.gate_pulses, signal_t.gate, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         newton_state, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state