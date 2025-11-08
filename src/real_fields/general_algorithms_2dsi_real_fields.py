from src.real_fields.base_classes_methods import RetrievePulses2DSIwithRealFields, RetrievePulsesRealFields
from src.twodsi.general_algorithms_2dsi import DifferentialEvolution as DifferentialEvolution2DSI, Evosax as Evosax2DSI, LSF as LSF2DSI, AutoDiff as AutoDiff2DSI
from src.utilities import MyNamespace, do_interpolation_1d
import jax



class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolution2DSI, RetrievePulses2DSIwithRealFields):
    """ The Differential Evolution Algorithm applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, DifferentialEvolution2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
        self._post_init()



class Evosax(RetrievePulsesRealFields, Evosax2DSI, RetrievePulses2DSIwithRealFields):
    """ The Evosax package applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, Evosax2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
        self._post_init()



class LSF(RetrievePulsesRealFields, LSF2DSI, RetrievePulses2DSIwithRealFields):
    """ The LSF Algorithm applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, LSF2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
        self._post_init()


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """
        Real fields LSF does not use make_pulse_f_from_individual()/make_pulse_t_from_individual().
        It stores the fields as complex values instead of amp/phase.
        This function interpolates the individuals onto time_big.
        """
        pulse_t_arr, gate_t_arr = super().get_pulses_from_population(population, measurement_info, descent_info)

        pulse_f_arr = self.fft(pulse_t_arr, measurement_info.sk, measurement_info.rn)
        pulse_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, pulse_f_arr)
        pulse_t_arr = self.ifft(pulse_f_arr, measurement_info.sk_big, measurement_info.rn_big)

        if measurement_info.doubleblind==True:
            gate_f_arr = self.fft(gate_t_arr, measurement_info.sk, measurement_info.rn)
            gate_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, gate_f_arr)
            gate_t_arr = self.ifft(gate_f_arr, measurement_info.sk_big, measurement_info.rn_big)
        else:
            gate_t_arr = pulse_t_arr

        return pulse_t_arr, gate_t_arr



    def convert_population(self, population, measurement_info, descent_info):
        """
        convert_population() does use make_pulse_f_from_individual()/make_pulse_t_from_individual().
        Thus the pulses are converted onto time_big. 
        The optimization is likely more efficient when the pulses are defined on time. Thus the interpolation.
        """
        population = super().convert_population(population, measurement_info, descent_info)
        pulse_f_arr = self.fft(population.pulse, measurement_info.sk_big, measurement_info.rn_big)
        pulse_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency, measurement_info.frequency_big, pulse_f_arr)
        pulse_t_arr = self.ifft(pulse_f_arr, measurement_info.sk, measurement_info.rn)

        if measurement_info.doubleblind==True:
            gate_f_arr = self.fft(population.gate, measurement_info.sk_big, measurement_info.rn_big)
            gate_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency, measurement_info.frequency_big, gate_f_arr)
            gate_t_arr = self.ifft(gate_f_arr, measurement_info.sk, measurement_info.rn)
        else:
            gate_t_arr = pulse_t_arr

        return MyNamespace(pulse=pulse_t_arr, gate=gate_t_arr)
    


    def make_population_bisection_search(self, E_arr, population, measurement_info, descent_info, pulse_or_gate):
        """
        For the generation of the signal field the pulses need to be interpolated onto time_big.
        """
        population = super().make_population_bisection_search(E_arr, population, measurement_info, descent_info, pulse_or_gate)

        pulse_f_arr = self.fft(population.pulse, measurement_info.sk, measurement_info.rn)
        pulse_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, pulse_f_arr)
        pulse_t_arr = self.ifft(pulse_f_arr, measurement_info.sk_big, measurement_info.rn_big)

        if measurement_info.doubleblind==True:
            gate_f_arr = self.fft(population.gate, measurement_info.sk, measurement_info.rn)
            gate_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, gate_f_arr)
            gate_t_arr = self.ifft(gate_f_arr, measurement_info.sk_big, measurement_info.rn_big)
        else:
            gate_t_arr = pulse_t_arr

        return MyNamespace(pulse=pulse_t_arr, gate=gate_t_arr)
    

    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info):
        """ Another interpolation onto time_big/frequency_big to calculate the signal field for the final trace. """
        pulse_t, gate_t, pulse_f, gate_f = super().post_process_get_pulse_and_gate(descent_state, measurement_info, descent_info)
        
        pulse_f = self.fft(pulse_t, measurement_info.sk, measurement_info.rn)
        pulse_f = do_interpolation_1d(measurement_info.frequency_big, measurement_info.frequency, pulse_f)
        pulse_t = self.ifft(pulse_f, measurement_info.sk_big, measurement_info.rn_big)

        if measurement_info.doubleblind==True:
            gate_f = self.fft(gate_t, measurement_info.sk, measurement_info.rn)
            gate_f = do_interpolation_1d(measurement_info.frequency_big, measurement_info.frequency, gate_f)
            gate_t = self.ifft(gate_f, measurement_info.sk_big, measurement_info.rn_big)
        else:
            gate_t = pulse_t

        return pulse_t, gate_t, pulse_f, gate_f



class AutoDiff(RetrievePulsesRealFields, AutoDiff2DSI, RetrievePulses2DSIwithRealFields):
    """ The Optimistix package applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, AutoDiff2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
        self._post_init()
