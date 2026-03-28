import jax.numpy as jnp

from equinox import tree_at

import refractiveindex

from pulsedjax.core.base_classes_methods import RetrievePulsesVAMPIRE
from pulsedjax.core.base_classic_algorithms import LSGPABASE, CPCGPABASE, GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE, LSFBASE

from pulsedjax.core.gradients.vampire_z_error_gradients import calculate_Z_gradient
from pulsedjax.core.hessians.vampire_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pulsedjax.core.hessians.pie_pseudo_hessian import PIE_get_pseudo_newton_direction

from pulsedjax.utilities import calculate_gate



class LSGPA(LSGPABASE, RetrievePulsesVAMPIRE):
    __doc__ = LSGPABASE.__doc__

    
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, **kwargs)
        
        assert self.doubleblind==False


class CPCGPA(CPCGPABASE, RetrievePulsesVAMPIRE):
    __doc__ = CPCGPABASE.__doc__

    
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, constraints=False, svd=False, antialias=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, constraints=constraints, svd=svd, antialias=antialias, **kwargs)

    
    def calculate_gate(self, gate_pulse, measurement_info):
        tau, nonlinear_method = measurement_info.tau_interferometer, measurement_info.nonlinear_method
        sk, rn, frequency, time = measurement_info.sk, measurement_info.rn, measurement_info.frequency, measurement_info.time
        
        # this will be wrong if i change apply phase to expect signals in f domain
        #gate_pulse_f = self.fft(gate_pulse, sk, rn)
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


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, signal_t.pulse_t, signal_t.gate_pulses, signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t, signal_t.gate_pulses, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, descent_state, measurement_info, 
                                                                         descent_info, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state







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


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, measured_trace, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for a given shift. """
        alpha = descent_info.alpha
        difference_signal_t = signal_t_new - signal_t.signal_t

        probe = signal_t.gate_shifted
        grad = -1*jnp.conjugate(probe)*difference_signal_t
        U = self.get_PIE_weights(probe, alpha, pie_method)
        return grad*U
    


    # def get_gate_probe_for_hessian(self, pulse_t, gate_pulse_shifted, nonlinear_method):
    #     pass


    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        
        """ Calculates the PIE newton direction for a population. """
        
        newton_direction_prev = getattr(local_or_global_state.newton, pulse_or_gate).newton_direction_prev
        probe = signal_t.gate_shifted

        descent_direction, newton_state = PIE_get_pseudo_newton_direction(grad, probe, signal_t.signal_f, tau_arr, measured_trace, newton_direction_prev, 
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


    def get_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, signal_t.pulse_t, signal_t.gate_pulses, signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        
        newton_state = local_or_global_state.newton
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t, signal_t.gate_pulses, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, 
                                                                         local_or_global_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state
    





class LSF(LSFBASE, RetrievePulsesVAMPIRE):
    __doc__ = LSFBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, **kwargs)

    

        
