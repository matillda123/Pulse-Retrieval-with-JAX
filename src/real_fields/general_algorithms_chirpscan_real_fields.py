from src.real_fields.base_classes_methods import RetrievePulsesCHIRPSCANwithRealFields, RetrievePulsesRealFields
from src.chirp_scan.general_algorithms_chirpscan import (DifferentialEvolution as DifferentialEvolutionCHIRPSCAN, Evosax as EvosaxCHIRPSCAN, 
                                          LSF as LSFCHIRPSCAN, AutoDiff as AutoDiffCHIRPSCAN)
from src.utilities import MyNamespace, do_interpolation_1d
import jax



class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolutionCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Differential Evolution Algorithm applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, DifferentialEvolutionCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
        self._post_init()



class Evosax(RetrievePulsesRealFields, EvosaxCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Evosax package applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, EvosaxCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
        self._post_init()



class LSF(RetrievePulsesRealFields, LSFCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The LSF Algorithm applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, LSFCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
        self._post_init()

    
    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """
        Real fields LSF does not use make_pulse_f_from_individual()/make_pulse_t_from_individual().
        It stores the fields as complex values instead of amp/phase.
        This function interpolates the individuals onto frequency_big.
        """
        pulse_f_arr, gate_f_arr = super().get_pulses_from_population(population, measurement_info, descent_info)

        pulse_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, pulse_f_arr)
        if measurement_info.doubleblind==True:
            gate_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, gate_f_arr)
        else:
            gate_f_arr = pulse_f_arr

        return pulse_f_arr, gate_f_arr



    def convert_population(self, population, measurement_info, descent_info):
        """
        convert_population() does use make_pulse_f_from_individual()/make_pulse_t_from_individual().
        Thus the pulses are converted onto frequency_big. 
        The optimization is likely more efficient when the pulses are defined on frequency. Thus the interpolation.
        """
        population = super().convert_population(population, measurement_info, descent_info)
        pulse_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency, measurement_info.frequency_big, population.pulse)

        if measurement_info.doubleblind==True:
            gate_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency, measurement_info.frequency_big, population.gate)
        else:
            gate_f_arr = pulse_f_arr

        return MyNamespace(pulse=pulse_f_arr, gate=gate_f_arr)
    


    def make_population_bisection_search(self, E_arr, population, measurement_info, descent_info, pulse_or_gate):
        """
        For the generation of the signal field the pulses need to be interpolated onto frequency_big.
        """
        population = super().make_population_bisection_search(E_arr, population, measurement_info, descent_info, pulse_or_gate)

        pulse_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, population.pulse)
        if measurement_info.doubleblind==True:
            gate_f_arr = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.frequency_big, measurement_info.frequency, population.gate)
        else:
            gate_f_arr = pulse_f_arr

        return MyNamespace(pulse=pulse_f_arr, gate=gate_f_arr)
    

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



class AutoDiff(RetrievePulsesRealFields, AutoDiffCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Optimistix package applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, AutoDiffCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
        self._post_init()
