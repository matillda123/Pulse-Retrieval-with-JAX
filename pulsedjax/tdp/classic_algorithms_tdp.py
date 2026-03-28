from equinox import tree_at

from pulsedjax.core.base_classes_methods import RetrievePulsesTDP
from pulsedjax.core.base_classic_algorithms import LSGPABASE, CPCGPABASE, GeneralizedProjectionBASE, COPRABASE, LSFBASE

from pulsedjax.core.gradients.tdp_z_error_gradients import calculate_Z_gradient
from pulsedjax.core.hessians.tdp_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error

from pulsedjax.frog import PtychographicIterativeEngine as PtychgraphicIterativeEngineFROG

from pulsedjax.utilities import calculate_gate




class LSGPA(LSGPABASE, RetrievePulsesTDP):
    __doc__ = LSGPABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, **kwargs)
        



class CPCGPA(CPCGPABASE, RetrievePulsesTDP):
    __doc__ = CPCGPABASE.__doc__

    def __init__(self, delay, frequency, trace, nonlinear_method, spectral_filter, cross_correlation=False, constraints=False, svd=False, antialias=False, **kwargs):
        super().__init__(delay, frequency, trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, constraints=constraints, svd=svd, antialias=antialias, **kwargs)
        

    
    def calculate_gate(self, gate_pulse, measurement_info):
        gate_pulse = self.apply_spectral_filter(gate_pulse, measurement_info.spectral_filter, measurement_info.sk, measurement_info.rn)
        return calculate_gate(gate_pulse, measurement_info.nonlinear_method)





class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulsesTDP):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, signal_t.pulse_t, signal_t.pulse_t_shifted, signal_t.gate_shifted, tau_arr, 
                                    measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr,
                                                                         descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state











class PtychographicIterativeEngine(RetrievePulsesTDP, PtychgraphicIterativeEngineFROG):
    __doc__ = PtychgraphicIterativeEngineFROG.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "Dont use interferometric with PIE. its not meant or made for that"







class COPRA(COPRABASE, RetrievePulsesTDP):
    __doc__ = COPRABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)




    def get_Z_gradient_individual(self, signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, signal_t.pulse_t, signal_t.pulse_t_shifted, 
                                    signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, 
                                                                         local_or_global_state, measurement_info, descent_info, 
                                                                         full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state








class LSF(LSFBASE, RetrievePulsesTDP):
    __doc__ = LSFBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)