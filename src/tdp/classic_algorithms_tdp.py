from equinox import tree_at

from src.core.base_classes_methods import RetrievePulsesTDP
from src.core.base_classic_algorithms import GeneralizedProjectionBASE, COPRABASE

from src.core.gradients.tdp_z_error_gradients import calculate_Z_gradient
from src.core.hessians.tdp_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error

from src.frog import PtychographicIterativeEngine as PtychgraphicIterativeEngineFROG





class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulsesTDP):
    """
    The Generalized Projection Algorithm for FROG.
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, tau_arr, 
                                    measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, descent_state.population.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         descent_state.newton, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent_direction and step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = self.fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = self.ifft(pulse_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse)
        return individual










class PtychographicIterativeEngine(RetrievePulsesTDP, PtychgraphicIterativeEngineFROG):
    """
    The Ptychographic Iterative Engine (PIE) for FROG.

    Attributes:
        pie_method (None, str): specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE. Where None indicates that the pure gradient is used.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, pie_method="rPIE", cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.ifrog==False, "Dont use ifrog with PIE. its not meant or made for that"

        self.pie_method=pie_method








class COPRA(COPRABASE, RetrievePulsesTDP):
    """
    The Common Pulse Retrieval Algorithm (COPRA) for FROG.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)



    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = self.ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual


    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.pulse_t_shifted, 
                                    signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """


        newton_state = local_or_global_state.newton
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, population.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         newton_state, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state

